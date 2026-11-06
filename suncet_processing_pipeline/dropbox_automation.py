"""Opt-in Dropbox delivery of snapshots from successful processing runs.

The processing hook freezes changed products; a separate worker transfers them.
No raw Level 0, CTDB, or failed/in-progress run is added to the queue.
"""
from __future__ import annotations

import argparse
import configparser
from contextlib import contextmanager
from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path
import shutil
import uuid

from . import rclone_public_data as copies
from .data_paths import get_data_root

PRODUCT_ROOTS = {'level0_5', 'level1', 'level2', 'level3', 'level4', 'trends'}


def _enabled(config: Path) -> bool:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(config)
    return parser.getboolean('automation', 'enabled', fallback=False)


def _state(task):
    copies._private_directory(task.state_directory)
    path = task.state_directory / 'automation'
    copies._private_directory(path)
    for name in ('pending', 'completed', 'manifests'):
        copies._private_directory(path / name)
    return path


def _products(task):
    """Inventory only explicitly approved product roots, never private manifests."""
    allowed = {p.as_posix() for p in copies._publication_roots(task.filter_file)}
    records = {}
    for name in sorted(PRODUCT_ROOTS & allowed):
        directory = task.local_path / name
        if directory.is_symlink():
            raise copies.RclonePublicError('Product root is a symlink')
        if not directory.exists():
            continue
        for path in directory.rglob('*'):
            relative = path.relative_to(task.local_path)
            if path.is_symlink():
                raise copies.RclonePublicError('Product tree contains a symlink')
            if (any(p.startswith('.') for p in relative.parts)
                    or path.name.endswith(('.partial', '.tmp')) or '.partial.' in path.name):
                continue
            if path.is_dir():
                continue
            info = path.stat()
            if not path.is_file():
                raise copies.RclonePublicError('Product tree contains a non-file')
            records[relative.as_posix()] = (info.st_size, info.st_mtime_ns,
                                          info.st_ctime_ns, info.st_ino)
    return records


class Completion:
    def __init__(self, task, run_id, run_kind):
        self.task, self.run_id, self.run_kind = task, run_id, run_kind
        self.state = _state(task)
        self.lock = (self.state / 'processing.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.before = _products(task)
        except BaseException:
            self.lock.close()
            raise

    def finish(self, success):
        try:
            if not success:
                return None
            after = _products(self.task)
            changed = sorted(p for p in after if after[p] != self.before.get(p))
            if not changed:
                return None
            root = self.task.local_path
            outbox = root / '.dropbox-outbox'
            if outbox.is_symlink():
                raise copies.RclonePublicError('Dropbox outbox is a symlink')
            copies._private_directory(outbox)
            identifier = uuid.uuid4().hex
            stage = outbox / identifier
            stage.mkdir(mode=0o700)
            records = []
            for name in changed:
                src, dst = root / name, stage / name
                before = copies._stable_publication_record(root, src)
                dst.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                shutil.copy2(src, dst)
                dst.chmod(0o600)
                if (copies._stable_publication_record(root, src) != before
                        or copies._stable_publication_record(stage, dst) != before):
                    raise copies.RclonePublicError('Product changed while freezing Dropbox delivery')
                records.append(before)
            # This minimal completion record contains no invocation, CTDB, host,
            # credentials, or raw telemetry paths from private run provenance.
            manifest_roots = {p.as_posix() for p in copies._publication_roots(self.task.filter_file)}
            if 'processing_manifests' in manifest_roots:
                copies._write_private_json(stage / 'processing_manifests' / f'dropbox_{identifier}.json', {
                    'schema_version': 1, 'status': 'succeeded', 'run_id': self.run_id,
                    'run_kind': self.run_kind, 'products': records,
                })
            manifest = self.state / 'manifests' / f'{identifier}.json'
            task = replace(self.task, local_path=stage)
            copies.create_publication_manifest(task, manifest, data_root=root)
            copies._write_private_json(self.state / 'pending' / f'{identifier}.json', {
                'schema_version': 1, 'id': identifier,
                'run_id': self.run_id, 'file_count': len(records),
            })
            return identifier
        finally:
            self.lock.close()


def begin_processing(run_id, run_kind, *, config=None):
    config = Path(config or copies.DEFAULT_CONFIG_PATH)
    if not _enabled(config):
        return None
    task = copies.load_task(config, 'push-products')
    if task.local_path != get_data_root().resolve():
        raise copies.RclonePublicError('Automated product task must use the managed data root')
    return Completion(task, run_id, run_kind)


@contextmanager
def processing_completion(run_id, run_kind, *, config=None):
    """Convenience context for processing entrypoints without run provenance."""
    completion = begin_processing(run_id, run_kind, config=config)
    try:
        yield
    except BaseException:
        if completion is not None:
            completion.finish(False)
        raise
    else:
        if completion is not None:
            completion.finish(True)


def drain(config=None, *, execute=False):
    config = Path(config or copies.DEFAULT_CONFIG_PATH)
    if not _enabled(config):
        raise copies.RclonePublicError('Dropbox automation is not enabled')
    task = copies.load_task(config, 'push-products')
    state = _state(task)
    root = get_data_root().resolve()
    done = 0
    with (state / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for pending in sorted((state / 'pending').glob('*.json')):
            request = json.loads(pending.read_text())
            identifier = pending.stem
            if (len(identifier) != 32 or any(c not in '0123456789abcdef' for c in identifier)
                    or request.get('id') != identifier or request.get('schema_version') != 1):
                raise copies.RclonePublicError('Invalid Dropbox queue record')
            stage = root / '.dropbox-outbox' / identifier
            if stage.is_symlink() or not stage.resolve().is_relative_to(root):
                raise copies.RclonePublicError('Unsafe Dropbox staging directory')
            _, receipt = copies.run_task(
                replace(task, local_path=stage), execute=execute, data_root=root,
                publication_manifest=state / 'manifests' / f'{identifier}.json',
            )
            if execute:
                copies._write_private_json(state / 'completed' / pending.name,
                    {**request, 'receipt': str(receipt)}, replace=True)
                pending.unlink()
                shutil.rmtree(stage)
                done += 1
    return done


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['push', 'pull-metadata'])
    parser.add_argument('--config', type=Path, default=copies.DEFAULT_CONFIG_PATH)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    os.umask(0o077)
    if args.command == 'push':
        print(json.dumps({'completed_deliveries': drain(args.config, execute=args.execute)}))
    else:
        task = copies.load_task(args.config, 'pull-metadata')
        _, receipt = copies.run_task(task, execute=args.execute)
        print(json.dumps({'task': 'pull-metadata', 'receipt': str(receipt)}))


if __name__ == '__main__':
    main()
