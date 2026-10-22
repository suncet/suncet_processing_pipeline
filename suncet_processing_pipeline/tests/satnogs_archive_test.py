"""Archive custody, crash recovery and source isolation; no live cloud writes."""
import base64
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest
from suncet_processing_pipeline.satnogs import archive as a, collect as c
from suncet_processing_pipeline.tests.satnogs_collect_test import config, FakeClient, page, record, NOW


class MemoryArchive:
    def __init__(self):
        self.objects = {}
        self.calls = []
        self.fail_at = None

    def ensure(self, key, body, content_type):
        self.calls.append(key)
        if len(self.calls) == self.fail_at:
            raise a.ArchiveError('offline')
        assert key.startswith('satnogs/')
        if key in self.objects:
            assert self.objects[key] == body
        self.objects[key] = body
        return {'VersionId': 'version1'}


@pytest.fixture
def archived_config(config):
    return replace(config, archive=a.ArchiveConfig('private-test', 'test', 'us-east-2'))


def test_upload_immediately_per_page_and_dedupe_frames_between_receptions(archived_config):
    uploader = MemoryArchive()
    class Client(FakeClient):
        def get(self, url):
            if self.urls:
                assert len(uploader.objects) == 2  # archived before second API GET
            return super().get(url)
    result = c.collect(archived_config, Client(page([record()], c.ENDPOINT+'?page=2'),
                       page([record(station_id=3)])), now=NOW, archiver=uploader)
    assert result['archived_receipts'] == 2
    assert len(uploader.objects) == 3  # identical frame shared, two provenance receipts
    manifest = json.loads(next(body for key, body in uploader.objects.items() if key.endswith('.json')))
    assert manifest['api_record']['observation_id'] == 2
    assert manifest['decoder_revision'].startswith('ctdb-2.0.5:')
    assert uploader.objects[manifest['frame_key']] == bytes.fromhex(record()['frame'])
    assert c.status(archived_config)['archive_pending'] == 0
    repeat = c.collect(archived_config, FakeClient(page([record()])), now=NOW, archiver=uploader)
    assert repeat['archived_receipts'] == 0 and len(uploader.calls) == 3


def test_upload_failure_keeps_local_receipt_and_checkpoint_recovers_before_network(archived_config):
    uploader = MemoryArchive()
    uploader.fail_at = 2
    with pytest.raises(a.ArchiveError):
        c.collect(archived_config, FakeClient(page([record()])), now=NOW, archiver=uploader)
    state = c.status(archived_config)
    assert state['archive_pending'] == 1 and 'last_success' not in state
    uploader.fail_at = None
    result = c.collect(archived_config, FakeClient(page()), now=NOW, archiver=uploader)
    assert result['archived_receipts'] == 1 and len(uploader.objects) == 2
    assert len([k for k in uploader.calls if k.endswith('.bin')]) == 1


def test_backfill_rejected_and_malformed_frames_preserves_source(archived_config):
    c.collect(replace(archived_config, archive=None), FakeClient(page([
        record(frame='0102'), record(frame='invalid')])), now=NOW)
    uploader = MemoryArchive()
    c.collect(archived_config, FakeClient(page()), now=NOW, archiver=uploader)
    assert len(uploader.objects) == 3
    assert b'\x01\x02' in uploader.objects.values()
    assert c.status(archived_config)['rejected'] == 2


def test_new_destination_is_not_mistaken_for_archived(archived_config):
    one, two = MemoryArchive(), MemoryArchive()
    c.collect(archived_config, FakeClient(page([record()])), now=NOW, archiver=one)
    changed = replace(archived_config, archive=replace(archived_config.archive, prefix='satnogs/v2/'))
    c.collect(changed, FakeClient(page()), now=NOW, archiver=two)
    assert len(two.objects) == 2
    assert all(k.startswith('satnogs/v2/') for k in two.objects)


@pytest.mark.parametrize('prefix', ['uhf/', '_test/', 'satnogs/../uhf/', '/satnogs/', 'satnogs//', 'satnogs'])
def test_config_refuses_other_sources(prefix):
    with pytest.raises(a.ArchiveError):
        a.ArchiveConfig.read(dict(bucket='b', profile='p', region='r', prefix=prefix))


def head(body):
    return dict(ContentLength=len(body), ChecksumSHA256=base64.b64encode(hashlib.sha256(body).digest()).decode(),
                ChecksumType='FULL_OBJECT', StorageClass='DEEP_ARCHIVE', VersionId='v1')


def test_real_adapter_conditional_put_and_head_verification(tmp_path, monkeypatch):
    body = b'test'
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command[2] == 'put-object':
            assert Path(command[command.index('--body')+1]).read_bytes() == body
            assert command[command.index('--if-none-match')+1] == '*'
            assert command[command.index('--storage-class')+1] == 'DEEP_ARCHIVE'
            return SimpleNamespace(returncode=1, stderr='An error occurred (PreconditionFailed)', stdout='')
        return SimpleNamespace(returncode=0, stderr='', stdout=json.dumps(head(body)))
    monkeypatch.setattr(a.subprocess,'run',run)
    assert a.S3Archive(a.ArchiveConfig('b','p','r'),tmp_path).ensure('satnogs/f.bin',body,'application/octet-stream')['VersionId']=='v1'
    assert len(calls)==2 and not list(tmp_path.glob('*.partial'))


@pytest.mark.parametrize('change', [dict(ContentLength=99), dict(ChecksumSHA256='bad'),
    dict(StorageClass='STANDARD'),dict(VersionId='null'),dict(ChecksumType='COMPOSITE')])
def test_archive_mismatch_is_not_accepted(tmp_path, change):
    uploader=a.S3Archive(a.ArchiveConfig('b','p','r'),tmp_path)
    uploader.call=lambda args,**kw: {**head(b'test'),**change}
    with pytest.raises(a.ArchiveError,match='verification'):
        uploader.ensure('satnogs/f.bin',b'test','application/octet-stream')


def test_denied_put_does_not_try_read_or_expose_aws_error(tmp_path,monkeypatch):
    monkeypatch.setattr(a.subprocess,'run',lambda *args,**kw: SimpleNamespace(
        returncode=1,stderr='AccessDenied private-bucket credential',stdout=''))
    with pytest.raises(a.ArchiveError) as error:
        a.S3Archive(a.ArchiveConfig('b','p','r'),tmp_path).ensure('satnogs/f.bin',b'hi','application/octet-stream')
    assert 'private-bucket' not in str(error.value)
