import json
from pathlib import Path

import pytest

from suncet_processing_pipeline import dropbox_automation as a
from suncet_processing_pipeline import rclone_public_data as c
from suncet_processing_pipeline.tests.rclone_public_data_test import _task_config


@pytest.fixture
def context(tmp_path, monkeypatch):
    root = tmp_path / 'data'
    root.mkdir()
    monkeypatch.setenv('suncet_data', str(root))
    monkeypatch.delenv('suncet_ctdb', raising=False)
    rules = tmp_path / 'products.filter'
    rules.write_text('+ /level0_5/**\n+ /level1/**\n+ /processing_manifests/**\n- /**\n')
    config = _task_config(tmp_path/'private', root, direction='push', filter_path=rules)
    with config.open('a') as f:
        f.write('\n[automation]\nenabled = true\n')
    return root, config


def put(root, name, body='example'):
    p = root/name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)
    return p


def test_disabled_hook_is_noop(tmp_path):
    assert a.begin_processing('run','test',config=tmp_path/'missing.ini') is None


def test_success_freezes_changed_products_only_and_omits_private_provenance(context):
    root, config = context
    put(root, 'level1/unchanged.fits')
    with a.processing_completion('run','test',config=config):
        put(root, 'level0/private.bin')
        put(root, 'level0_5/image.fits', 'complete')
        put(root, 'level1/image.fits.partial', 'incomplete')
        put(root, 'processing_manifests/private.json', 'private invocation')
    task=c.load_task(config,'push-products')
    pending=list((a._state(task)/'pending').glob('*.json'))
    assert len(pending)==1
    stage=root/'.dropbox-outbox'/pending[0].stem
    assert (stage/'level0_5/image.fits').read_text()=='complete'
    assert not (stage/'level0').exists()
    assert not (stage/'level1').exists()
    manifest=list((stage/'processing_manifests').glob('*.json'))
    assert len(manifest)==1 and 'private invocation' not in manifest[0].read_text()
    put(root, 'level0_5/image.fits', 'next run')
    assert (stage/'level0_5/image.fits').read_text()=='complete'


def test_failure_and_noop_do_not_enqueue(context):
    root,config=context
    with pytest.raises(RuntimeError):
        with a.processing_completion('bad','test',config=config):
            put(root,'level0_5/incomplete.fits')
            raise RuntimeError('processing failed')
    with a.processing_completion('noop','test',config=config):
        pass
    assert not list((a._state(c.load_task(config,'push-products'))/'pending').glob('*.json'))


def test_worker_retains_failed_delivery_and_retries_exact_snapshot(context, monkeypatch):
    root, config=context
    with a.processing_completion('run','test',config=config):
        put(root,'level0_5/image.fits','data')
    def failed(*args,**kwargs): raise c.RclonePublicError('offline')
    monkeypatch.setattr(c,'run_task',failed)
    with pytest.raises(c.RclonePublicError,match='offline'): a.drain(config,execute=True)
    state=a._state(c.load_task(config,'push-products'))
    assert len(list((state/'pending').glob('*.json')))==1
    def copied(task,**kwargs):
        assert kwargs['execute']
        payload=c._load_publication_manifest(kwargs['publication_manifest'],task,root)
        c._verify_publication_manifest(payload,task)
        assert (task.local_path/'level0_5/image.fits').read_text()=='data'
        return Path('log'),Path('receipt')
    monkeypatch.setattr(c,'run_task',copied)
    assert a.drain(config,execute=True)==1
    assert a.drain(config,execute=True)==0
    assert len(list((state/'completed').glob('*.json')))==1
    assert not list((root/'.dropbox-outbox').iterdir())


def test_symlink_and_concurrent_processing_are_refused(context):
    root,config=context
    first=a.begin_processing('first','test',config=config)
    try:
        with pytest.raises(BlockingIOError): a.begin_processing('second','test',config=config)
    finally: first.finish(False)
    (root/'level0_5').symlink_to(root/'level0')
    with pytest.raises(c.RclonePublicError,match='symlink'):
        a.begin_processing('third','test',config=config)


def test_corrupt_snapshot_is_rejected_before_upload(context,monkeypatch):
    root,config=context
    with a.processing_completion('run','test',config=config):
        put(root,'level0_5/image.fits','data')
    stage=next((root/'.dropbox-outbox').iterdir())
    (stage/'level0_5/image.fits').write_text('corrupt')
    with pytest.raises(c.RclonePublicError,match='preflight'):
        a.drain(config,execute=True)
    assert list((a._state(c.load_task(config,'push-products'))/'pending').glob('*.json'))


def test_provenance_completion_captures_products_outside_input_folder(context, monkeypatch):
    from suncet_processing_pipeline.run_provenance import ProcessingRunProvenance
    root, config = context
    monkeypatch.setattr(c, 'DEFAULT_CONFIG_PATH', config)
    with ProcessingRunProvenance(data_root=root/'level0'/'station', run_kind='integration'):
        put(root, 'level0_5/from_input.fits', 'finished')
    state = a._state(c.load_task(config,'push-products'))
    assert len(list((state/'pending').glob('*.json'))) == 1
    with pytest.raises(ValueError):
        with ProcessingRunProvenance(data_root=root/'level0'/'station', run_kind='integration'):
            put(root, 'level0_5/failed.fits', 'not finished')
            raise ValueError('failure')
    assert len(list((state/'pending').glob('*.json'))) == 1
