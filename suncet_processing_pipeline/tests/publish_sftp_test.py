"""Tests for safe SFTP publication without contacting an external server."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

import pytest

from ..publish_sftp import (
    OpenSftpTransport,
    PublicationConflictError,
    PublicationError,
    SftpPublisher,
    _quote_sftp,
    _write_transfer_log,
    collect_publication_files,
)


class FakeSftpTransport:
    def __init__(self):
        self.files: dict[PurePosixPath, bytes] = {}
        self.directories = {PurePosixPath("/")}
        self.put_counts: dict[PurePosixPath, int] = {}
        self.corrupt_staged_downloads = 0

    def exists(self, remote_path):
        return remote_path in self.files or remote_path in self.directories

    def mkdirs(self, remote_directory):
        current = PurePosixPath("/")
        for part in remote_directory.parts[1:]:
            current /= part
            self.directories.add(current)

    def put(self, local_path, remote_path):
        self.files[remote_path] = local_path.read_bytes()
        self.put_counts[remote_path] = self.put_counts.get(remote_path, 0) + 1

    def get(self, remote_path, local_path):
        data = self.files[remote_path]
        if (
            self.corrupt_staged_downloads
            and ".partial." in str(remote_path)
            and ".sha256.partial." not in str(remote_path)
        ):
            self.corrupt_staged_downloads -= 1
            data = b"corrupt readback"
        local_path.write_bytes(data)

    def rename(self, source, destination):
        if destination in self.files:
            raise PublicationError(f"destination already exists: {destination}")
        self.files[destination] = self.files.pop(source)

    def remove(self, remote_path):
        self.files.pop(remote_path, None)


def _publisher(transport):
    return SftpPublisher(
        transport,
        remote_root="/suncet_data",
        retries=2,
        retry_delay_seconds=0,
    )


def test_new_file_is_staged_read_back_and_published_with_checksum(tmp_path):
    source = tmp_path / "product.fits"
    source.write_bytes(b"finalized SunCET product")
    transport = FakeSftpTransport()

    result = _publisher(transport).publish_file(
        source, Path("2027/level1/product.fits")
    )

    final = PurePosixPath("/suncet_data/2027/level1/product.fits")
    sidecar = PurePosixPath(f"{final}.sha256")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    assert result.status == "published"
    assert result.sha256 == expected
    assert transport.files[final] == source.read_bytes()
    assert transport.files[sidecar] == f"{expected}  product.fits\n".encode()
    assert not any(".partial." in str(path) for path in transport.files)


def test_matching_remote_file_and_sidecar_are_idempotently_skipped(tmp_path):
    source = tmp_path / "product.fits"
    source.write_bytes(b"same product")
    transport = FakeSftpTransport()
    publisher = _publisher(transport)

    first = publisher.publish_file(source, Path("2027/product.fits"))
    second = publisher.publish_file(source, Path("2027/product.fits"))

    assert first.status == "published"
    assert second.status == "unchanged"
    staged = next(path for path in transport.put_counts if ".partial." in str(path))
    assert transport.put_counts[staged] == 1


def test_matching_sidecar_does_not_hide_corrupt_remote_file(tmp_path):
    source = tmp_path / "product.fits"
    source.write_bytes(b"correct product")
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    transport = FakeSftpTransport()
    final = PurePosixPath("/suncet_data/2027/product.fits")
    transport.files[final] = b"corrupt remote product"
    transport.files[PurePosixPath(f"{final}.sha256")] = (
        f"{checksum}  product.fits\n".encode()
    )

    with pytest.raises(PublicationConflictError, match="checksum sidecar"):
        _publisher(transport).publish_file(source, Path("2027/product.fits"))


def test_existing_matching_file_without_sidecar_is_adopted_after_readback(tmp_path):
    source = tmp_path / "product.fits"
    source.write_bytes(b"existing product")
    transport = FakeSftpTransport()
    final = PurePosixPath("/suncet_data/2027/product.fits")
    transport.files[final] = source.read_bytes()

    result = _publisher(transport).publish_file(source, Path("2027/product.fits"))

    assert result.status == "adopted"
    assert PurePosixPath(f"{final}.sha256") in transport.files


def test_different_remote_content_is_never_overwritten(tmp_path):
    source = tmp_path / "product.fits"
    source.write_bytes(b"new product")
    transport = FakeSftpTransport()
    final = PurePosixPath("/suncet_data/2027/product.fits")
    transport.files[final] = b"older different product"

    with pytest.raises(PublicationConflictError, match="refusing to overwrite"):
        _publisher(transport).publish_file(source, Path("2027/product.fits"))

    assert transport.files[final] == b"older different product"


def test_corrupt_staged_readback_is_retried_before_finalization(tmp_path):
    source = tmp_path / "product.fits"
    source.write_bytes(b"verified on the second readback")
    transport = FakeSftpTransport()
    transport.corrupt_staged_downloads = 1

    result = _publisher(transport).publish_file(source, Path("2027/product.fits"))

    assert result.status == "published"
    staged = next(path for path in transport.put_counts if ".partial." in str(path))
    assert transport.put_counts[staged] == 2


def test_collect_files_maps_from_explicit_local_base_and_skips_temporary_files(
    tmp_path,
):
    data_root = tmp_path / "data"
    local_base = data_root / "level1"
    product_dir = local_base / "2027"
    product_dir.mkdir(parents=True)
    product = product_dir / "product.fits"
    product.write_bytes(b"product")
    (product_dir / "unfinished.fits.partial").write_bytes(b"partial")
    transfer_log = local_base / "transfer_logs" / "old.json"
    transfer_log.parent.mkdir()
    transfer_log.write_text("{}", encoding="utf-8")
    transfer_stage = local_base / "transfer_staging" / "readback.bin"
    transfer_stage.parent.mkdir()
    transfer_stage.write_bytes(b"staging")

    selected = collect_publication_files(
        [str(local_base)], data_root=data_root, local_base=local_base
    )

    assert selected == [(product, Path("2027/product.fits"))]


def test_collect_files_rejects_sources_outside_suncet_data(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    outside = tmp_path / "private.bin"
    outside.write_bytes(b"private")

    with pytest.raises(PublicationError, match="outside"):
        collect_publication_files(
            [str(outside)], data_root=data_root, local_base=data_root
        )


def test_collect_files_rejects_explicit_symlink(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    product = data_root / "product.fits"
    product.write_bytes(b"product")
    link = data_root / "linked.fits"
    link.symlink_to(product)

    with pytest.raises(PublicationError, match="symlink"):
        collect_publication_files(
            [str(link)], data_root=data_root, local_base=data_root
        )


def test_sftp_batch_quoting_handles_spaces_and_rejects_newlines():
    assert _quote_sftp("/suncet_data/file name.fits") == '"/suncet_data/file name.fits"'
    with pytest.raises(PublicationError, match="control character"):
        _quote_sftp("/suncet_data/bad\npath")


def test_sftp_transport_rejects_nonpositive_timeouts():
    with pytest.raises(PublicationError, match="timeouts must be positive"):
        OpenSftpTransport("lasp-sfs", command_timeout_seconds=0)


def test_transfer_log_is_atomic_json(tmp_path):
    payload = {"status": "succeeded", "items": [{"status": "published"}]}

    path = _write_transfer_log(tmp_path, payload)

    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert not path.with_suffix(".json.tmp").exists()
