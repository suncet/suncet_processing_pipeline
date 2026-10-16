"""Incrementally process new telemetry from the standard Level 0 routes.

Each raw delivery is treated as an independent unit so files already recorded in a
route's state ledger can be skipped before their potentially multi-gigabyte contents
are read.  The standard Level 0 tree maps Leafspace to X-band framing and the LASP,
Fairbanks, and SatNOGS ground-station folders to UHF framing.  An explicit folder and
input mode can also be supplied for one-off processing.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb

from suncet_processing_pipeline.config_parser import Config
from suncet_processing_pipeline.data_paths import data_path
from suncet_processing_pipeline.make_level0_5 import (
    DEFAULT_HARDLINE_CCSDS_PREFIX,
    DEFAULT_UHF_SUBDIR_BASENAME,
    DEFAULT_XBAND_PREFIX,
    INPUT_MODE_AUTO,
    INPUT_MODE_CCSDS,
    INPUT_MODE_COMBINED,
    INPUT_MODE_HARDLINE,
    INPUT_MODE_UHF,
    INPUT_MODE_XBAND,
    _build_argument_parser,
    discover_prefixed_binary_files,
    discover_uhf_binary_files,
    main as run_level0_5,
    resolve_config_data_folder,
)
from suncet_processing_pipeline.make_telemetry_file import TelemetryProcessor
from suncet_processing_pipeline.run_provenance import sha256_file


STATE_SCHEMA_VERSION = 1
DEFAULT_LEVEL0_FOLDER = ("level0",)
LEGACY_XBAND_STATE_BASENAME = "xband_incremental_state.json"
STANDARD_XBAND_PREFIX = "xband"
STANDARD_ROUTE_MODES = {
    "leafspace_ground_stations": INPUT_MODE_XBAND,
    "fairbanks_ground_station": INPUT_MODE_UHF,
    "lasp_ground_station": INPUT_MODE_UHF,
    "satnogs": INPUT_MODE_UHF,
}
_INTERNAL_ARGUMENTS = {
    "append_csie_inventory",
    "duckdb_source_key",
    "folder",
    "input_file",
    "prefix",
}


@dataclass(frozen=True)
class RouteSpec:
    """One independently tracked Level 0 delivery route."""

    route_id: str
    folder: Path
    input_mode: str
    prefix: str


def _default_prefix(input_mode: str) -> str:
    return (
        STANDARD_XBAND_PREFIX
        if input_mode == INPUT_MODE_XBAND
        else DEFAULT_HARDLINE_CCSDS_PREFIX
    )


def _route_state_basename(route_id: str) -> str:
    safe = "".join(character if character.isalnum() else "_" for character in route_id)
    return f"incremental_level0_{safe}_state.json"


def _safe_source_key(relative_path: str) -> str:
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _processing_signature(config: Config, args: object, config_path: Path) -> str:
    arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in _INTERNAL_ARGUMENTS
    }
    code_root = Path(__file__).resolve().parent
    code_hashes = {
        name: sha256_file(code_root / name)
        for name in ("make_level0_5.py", "make_telemetry_file.py", "incremental_level0.py")
    }
    payload = {
        "pipeline_version": str(config.version_pipeline),
        "bus_ctdb_version": str(config.version_bus),
        "csie_ctdb_version": str(config.version_csie),
        "dsps_ctdb_version": str(config.version_dsps),
        "config_sha256": sha256_file(config_path),
        "arguments": arguments,
        "code_sha256": code_hashes,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_state(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read X-band incremental state {path}: {exc}") from exc
    if payload.get("schema_version") != STATE_SCHEMA_VERSION:
        return None
    return payload


def _manifest_matches_current_settings(
    payload: dict[str, object],
    config: Config,
    input_mode: str,
) -> bool:
    if payload.get("status") != "succeeded":
        return False
    invocation = payload.get("invocation")
    arguments = invocation.get("arguments", {}) if isinstance(invocation, dict) else {}
    if not isinstance(arguments, dict):
        return False
    if arguments.get("input_mode") != input_mode:
        return False
    if arguments.get("skip_duckdb") is True:
        return False
    resolved = payload.get("resolved_configuration", {})
    if not isinstance(resolved, dict):
        return False
    expected = {
        "version_pipeline": str(config.version_pipeline),
        "version_bus": str(config.version_bus),
        "version_csie": str(config.version_csie),
        "version_dsps": str(config.version_dsps),
    }
    return all(str(resolved.get(name)) == value for name, value in expected.items())


def _bootstrap_entries_from_manifests(
    folder: Path,
    current_paths: dict[str, Path],
    config: Config,
    input_mode: str = INPUT_MODE_XBAND,
) -> dict[str, dict[str, object]]:
    """Recover every current input proven complete by a successful manifest."""
    manifest_dir = folder / "processing_manifests"
    recovered: dict[str, dict[str, object]] = {}
    current_hashes: dict[str, str] = {}

    def current_sha256(relative: str, path: Path) -> str:
        if relative not in current_hashes:
            current_hashes[relative] = sha256_file(path)
        return current_hashes[relative]

    for manifest_path in sorted(manifest_dir.glob("processing_run_*.json"), reverse=True):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not _manifest_matches_current_settings(payload, config, input_mode):
            continue
        invocation = payload.get("invocation", {})
        manifest_arguments = (
            invocation.get("arguments", {}) if isinstance(invocation, dict) else {}
        )
        manifest_source_key = (
            str(manifest_arguments.get("duckdb_source_key", ""))
            if isinstance(manifest_arguments, dict)
            else ""
        )
        manifest_inputs = payload.get("inputs", [])
        if not isinstance(manifest_inputs, list):
            continue
        candidate: dict[str, dict[str, object]] = {}
        used_current_paths: set[str] = set()
        valid = True
        for record in manifest_inputs:
            if not isinstance(record, dict):
                valid = False
                break
            relative = str(record.get("path", ""))
            path = current_paths.get(relative)
            if path is None:
                same_size = [
                    (current_relative, current_path)
                    for current_relative, current_path in current_paths.items()
                    if current_relative not in used_current_paths
                    and current_path.stat().st_size == int(record.get("size_bytes", -1))
                ]
                if len(same_size) == 1:
                    renamed_relative, renamed_path = same_size[0]
                    if current_sha256(renamed_relative, renamed_path) == str(
                        record.get("sha256", "")
                    ):
                        relative, path = renamed_relative, renamed_path
            if path is None:
                valid = False
                break
            stat = path.stat()
            recorded_sha256 = str(record.get("sha256", ""))
            if (
                int(record.get("size_bytes", -1)) != stat.st_size
                or not recorded_sha256
                or current_sha256(relative, path) != recorded_sha256
            ):
                valid = False
                break
            candidate[relative] = {
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": recorded_sha256,
                "source_key": (
                    manifest_source_key
                    if manifest_source_key and len(manifest_inputs) == 1
                    else _safe_source_key(relative)
                ),
                "storage": "merged_baseline",
            }
            used_current_paths.add(relative)
        if valid:
            for relative, entry in candidate.items():
                recovered.setdefault(relative, entry)
        if len(recovered) == len(current_paths):
            break
    return recovered


def _snapshot_after_success(
    path: Path,
    relative: str,
    source_key: str | None = None,
) -> dict[str, object]:
    stat = path.stat()
    return {
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
        "source_key": source_key or _safe_source_key(relative),
        "storage": "per_file",
    }


def _plan_incremental_inputs(
    entries: dict[str, dict[str, object]],
    current_paths: dict[str, Path],
    *,
    force_rebuild: bool,
) -> tuple[dict[str, dict[str, object]], list[str], list[str], list[str]]:
    """Select only new or replaced captures unless a rebuild was requested.

    Processing settings and code revisions deliberately do not enter into this
    decision.  They describe how the *next* capture is processed; historical raw
    captures are expensive and are only replayed after an explicit
    ``--rebuild`` request.
    """
    removed = sorted(set(entries) - set(current_paths))
    active_entries = {
        relative: entry
        for relative, entry in entries.items()
        if relative in current_paths
    }
    if force_rebuild:
        return {}, sorted(current_paths), [], removed

    changed: list[str] = []
    for relative, entry in active_entries.items():
        path = current_paths[relative]
        stat = path.stat()
        if (
            int(entry.get("size_bytes", -1)) == stat.st_size
            and int(entry.get("mtime_ns", -1)) == stat.st_mtime_ns
        ):
            continue
        if str(entry.get("sha256", "")) == sha256_file(path):
            entry["size_bytes"] = stat.st_size
            entry["mtime_ns"] = stat.st_mtime_ns
            continue
        changed.append(relative)

    new = set(current_paths) - set(active_entries)
    pending = sorted(new | set(changed))
    return active_entries, pending, changed, removed


def _database_processor(config: Config, args: object) -> TelemetryProcessor:
    return TelemetryProcessor(
        version=config.version_pipeline,
        database_path=args.telemetry_database,
        software_versions={
            "bus_ctdb_version": config.version_bus,
            "csie_ctdb_version": config.version_csie,
            "dsps_ctdb_version": config.version_dsps,
        },
    )


def _remove_superseded_database_sources(
    folder: Path,
    config: Config,
    args: object,
    current_source_keys: set[str],
    input_mode: str,
) -> None:
    """Remove stale per-file identities for one route after a successful rebuild."""
    if args.skip_duckdb:
        return
    processor = _database_processor(config, args)
    database_path = processor.database_path
    if not database_path.is_file():
        return
    decoded_root = str((folder / args.decoded_dir_name).resolve()) + os.sep
    with duckdb.connect(str(database_path)) as connection:
        rows = connection.execute(
            "SELECT ingestion_id, table_name, source_path FROM _ingestions "
            "WHERE starts_with(source_path, ?)",
            [decoded_root],
        ).fetchall()
        stale = []
        for ingestion_id, table_name, source_path in rows:
            marker = f"__{input_mode}_"
            source_text = str(source_path)
            if marker not in source_text:
                continue
            keep = any(f"{marker}{key}.csv" in source_text for key in current_source_keys)
            if not keep:
                stale.append((ingestion_id, table_name))
        if not stale:
            return
        connection.execute("BEGIN")
        try:
            for ingestion_id, table_name in stale:
                safe_table = str(table_name).replace('"', '""')
                connection.execute(
                    f'DELETE FROM "{safe_table}" WHERE _ingestion_id = ?',
                    [ingestion_id],
                )
                connection.execute(
                    "DELETE FROM _ingestions WHERE ingestion_id = ?",
                    [ingestion_id],
                )
            connection.execute("COMMIT")
        except Exception:
            try:
                connection.execute("ROLLBACK")
            except duckdb.TransactionException:
                pass
            raise
    print(f"Removed {len(stale):,} superseded merged/stale DuckDB ingestion(s).")


def _has_option(arguments: list[str], option: str) -> bool:
    return option in arguments or any(
        argument.startswith(f"{option}=") for argument in arguments
    )


def operational_arguments(argv: list[str]) -> list[str]:
    """Apply the standard Level 0 root while preserving explicit overrides.

    A first positional argument remains a convenience alias for ``--folder`` so
    the repository's ``./process_new_data /one/off/folder`` workflow is concise.
    """
    arguments = list(argv)
    if arguments and not arguments[0].startswith("-"):
        arguments[0:1] = ["--folder", arguments[0]]
    help_requested = any(argument in {"-h", "--help"} for argument in arguments)
    if not _has_option(arguments, "--folder") and not help_requested:
        arguments[0:0] = [
            "--folder",
            str(data_path(*DEFAULT_LEVEL0_FOLDER)),
        ]
    return arguments


def operational_main(argv: list[str] | None = None) -> None:
    """Run incremental processing across the standard Level 0 delivery routes."""
    main(operational_arguments(list(argv) if argv is not None else sys.argv[1:]))


def _build_operational_parser():
    parser = _build_argument_parser()
    parser.description = (
        "Incrementally process new or replaced telemetry across standard Level 0 "
        "ground-station delivery routes."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Replay every current input in the selected route(s).",
    )
    return parser


def _discover_route_inputs(route: RouteSpec) -> list[Path]:
    if route.input_mode == INPUT_MODE_UHF:
        return discover_uhf_binary_files(route.folder, route.prefix)
    return discover_prefixed_binary_files(route.folder, route.prefix)


def _detected_routes(folder: Path, args: object) -> list[RouteSpec]:
    """Return all raw-source formats found in one nonstandard folder."""
    requested_mode = (
        INPUT_MODE_CCSDS if args.input_mode == INPUT_MODE_HARDLINE else args.input_mode
    )
    if requested_mode == INPUT_MODE_COMBINED:
        requested_mode = INPUT_MODE_AUTO
    if requested_mode != INPUT_MODE_AUTO:
        prefix = args.prefix or _default_prefix(requested_mode)
        route = RouteSpec(requested_mode, folder, requested_mode, prefix)
        return [route] if _discover_route_inputs(route) else []
    if args.prefix is not None:
        inferred = (
            INPUT_MODE_CCSDS
            if args.prefix.startswith(DEFAULT_HARDLINE_CCSDS_PREFIX)
            else INPUT_MODE_XBAND
        )
        route = RouteSpec(inferred, folder, inferred, args.prefix)
        return [route] if _discover_route_inputs(route) else []

    routes: list[RouteSpec] = []
    xband = RouteSpec("xband", folder, INPUT_MODE_XBAND, STANDARD_XBAND_PREFIX)
    if _discover_route_inputs(xband):
        routes.append(xband)
    hardline = RouteSpec(
        "hardline", folder, INPUT_MODE_CCSDS, DEFAULT_HARDLINE_CCSDS_PREFIX
    )
    if _discover_route_inputs(hardline):
        routes.append(hardline)
    if (folder / DEFAULT_UHF_SUBDIR_BASENAME).is_dir():
        uhf = RouteSpec("uhf", folder, INPUT_MODE_UHF, DEFAULT_HARDLINE_CCSDS_PREFIX)
        if _discover_route_inputs(uhf):
            routes.append(uhf)
    return routes


def discover_level0_routes(folder: Path, args: object) -> list[RouteSpec]:
    """Map the standard Level 0 station folders, or inspect one override folder."""
    if folder.name != "level0":
        standard_mode = STANDARD_ROUTE_MODES.get(folder.name)
        if standard_mode is not None and args.input_mode == INPUT_MODE_AUTO:
            route = RouteSpec(
                folder.name,
                folder,
                standard_mode,
                args.prefix or _default_prefix(standard_mode),
            )
            return [route] if _discover_route_inputs(route) else []
        return _detected_routes(folder, args)

    if args.input_mode != INPUT_MODE_AUTO or args.prefix is not None:
        raise ValueError(
            "--input-mode and --prefix may only override one delivery folder, not "
            "the complete Level 0 root"
        )
    routes: list[RouteSpec] = []
    for child in sorted((path for path in folder.iterdir() if path.is_dir()), key=lambda p: p.name):
        standard_mode = STANDARD_ROUTE_MODES.get(child.name)
        if standard_mode is not None:
            route = RouteSpec(
                child.name,
                child,
                standard_mode,
                _default_prefix(standard_mode),
            )
            if _discover_route_inputs(route):
                routes.append(route)
            continue
        routes.extend(
            RouteSpec(
                f"{child.name}_{route.route_id}",
                route.folder,
                route.input_mode,
                route.prefix,
            )
            for route in _detected_routes(child, args)
        )
    return routes


def _strip_route_arguments(arguments: list[str]) -> list[str]:
    """Remove arguments that this wrapper sets independently for every route."""
    options_with_values = {
        "--folder",
        "--input-mode",
        "--prefix",
        "--input-file",
        "--duckdb-source-key",
    }
    wrapper_flags = {"--append-csie-inventory", "--rebuild"}
    cleaned: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if any(argument.startswith(f"{option}=") for option in options_with_values):
            index += 1
            continue
        if argument in options_with_values:
            index += 2
            continue
        if argument in wrapper_flags:
            index += 1
            continue
        cleaned.append(argument)
        index += 1
    return cleaned


def _process_route(
    route: RouteSpec,
    *,
    base_argv: list[str],
    config: Config,
    config_path: Path,
    force_rebuild: bool,
) -> tuple[int, int]:
    route_argv = [
        *base_argv,
        "--folder",
        str(route.folder),
        "--input-mode",
        route.input_mode,
        "--prefix",
        route.prefix,
    ]
    args = _build_argument_parser().parse_args(route_argv)
    input_paths = _discover_route_inputs(route)
    current_paths = {
        str(path.relative_to(route.folder)): path for path in input_paths
    }
    manifest_dir = route.folder / "processing_manifests"
    state_path = manifest_dir / _route_state_basename(route.route_id)
    legacy_state_path = manifest_dir / LEGACY_XBAND_STATE_BASENAME
    if (
        route.input_mode == INPUT_MODE_XBAND
        and not state_path.exists()
        and legacy_state_path.exists()
    ):
        state_path = legacy_state_path
    signature = _processing_signature(config, args, config_path)
    state = _read_state(state_path)
    entries: dict[str, dict[str, object]]
    if state is None:
        entries = _bootstrap_entries_from_manifests(
            route.folder, current_paths, config, route.input_mode
        )
        if entries:
            print(
                f"Adopted {len(entries):,} unchanged input file(s) from the latest "
                "successful Level 0.5 provenance manifest."
            )
    else:
        raw_entries = state.get("files", {})
        entries = dict(raw_entries) if isinstance(raw_entries, dict) else {}
        unrecorded_paths = {
            relative: path
            for relative, path in current_paths.items()
            if relative not in entries
        }
        recovered = _bootstrap_entries_from_manifests(
            route.folder,
            unrecorded_paths,
            config,
            route.input_mode,
        )
        if recovered:
            entries.update(recovered)
            print(
                f"Recovered {len(recovered):,} completed input file(s) from "
                "successful Level 0.5 provenance manifests."
            )

    signature_changed = (
        state is not None and state.get("processing_signature") != signature
    )
    entries, pending, changed_paths, removed_paths = _plan_incremental_inputs(
        entries,
        current_paths,
        force_rebuild=force_rebuild,
    )
    if force_rebuild:
        print(f"Rebuilding every {route.route_id} input: requested by --rebuild.")
    elif signature_changed:
        print(
            "Pipeline/configuration settings changed; keeping completed "
            "inputs and applying the current settings only to new inputs. Use "
            "--rebuild to replay all historical inputs."
        )
    if removed_paths:
        print(
            f"Forgot {len(removed_paths):,} recorded input(s) no longer present in "
            "the folder; existing DuckDB rows were left unchanged."
        )
    if changed_paths and not force_rebuild:
        print(
            f"Detected {len(changed_paths):,} previously recorded input(s) with "
            "replaced contents; only those inputs will be reprocessed."
        )

    if not pending:
        print(
            f"{route.route_id} inputs are already processed: "
            f"{len(current_paths):,} unchanged "
            "file(s), 0 files read or decoded."
        )
        _atomic_write_json(
            state_path,
            {
                "schema_version": STATE_SCHEMA_VERSION,
                "processing_signature": signature,
                "files": entries,
            },
        )
        return 0, len(entries)

    print(
        f"Incremental {route.route_id} plan: "
        f"{len(entries) - len(changed_paths):,} unchanged "
        "file(s) skipped; "
        f"{len(pending):,} file(s) to process independently."
    )
    for relative in pending:
        path = current_paths[relative]
        # Preserve the original stable DuckDB identity when a completed capture
        # was renamed or its contents were replaced under the same ledger entry.
        source_key = str(entries.get(relative, {}).get("source_key", ""))
        if not source_key:
            source_key = _safe_source_key(relative)
        print(f"\nProcessing pending {route.route_id} input: {relative}")
        run_level0_5(
            [
                *route_argv,
                "--input-file",
                str(path),
                "--duckdb-source-key",
                source_key,
                "--append-csie-inventory",
            ]
        )
        entries[relative] = _snapshot_after_success(path, relative, source_key)
        _atomic_write_json(
            state_path,
            {
                "schema_version": STATE_SCHEMA_VERSION,
                "processing_signature": signature,
                "files": entries,
            },
        )

    if force_rebuild:
        _remove_superseded_database_sources(
            route.folder,
            config,
            args,
            {str(entry["source_key"]) for entry in entries.values()},
            route.input_mode,
        )
    print(
        f"Incremental {route.route_id} processing complete: "
        f"{len(pending):,} file(s) processed, "
        f"{len(entries):,} total file(s) recorded."
    )
    return len(pending), len(entries)


def main(argv: list[str] | None = None) -> None:
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    parser = _build_operational_parser()
    args = parser.parse_args(argv_list)
    force_rebuild = args.rebuild
    if args.input_file:
        raise ValueError(
            "process_new_data selects individual --input-file values automatically"
        )

    config_path = Path(args.config).expanduser().resolve()
    config = Config(str(config_path))
    folder = (
        Path(args.folder).expanduser().resolve()
        if args.folder is not None
        else resolve_config_data_folder(config)
    )
    if not folder.is_dir():
        raise FileNotFoundError(f"Level 0 delivery folder does not exist: {folder}")
    routes = discover_level0_routes(folder, args)
    if not routes:
        print(f"No unambiguous Level 0 telemetry inputs found under {folder}.")
        return

    print(f"Level 0 root: {folder}")
    print("Delivery routes:")
    for route in routes:
        print(
            f"  {route.route_id}: {route.input_mode}, "
            f"{len(_discover_route_inputs(route)):,} file(s) under {route.folder}"
        )

    base_argv = _strip_route_arguments(argv_list)
    processed = 0
    recorded = 0
    for route in routes:
        route_processed, route_recorded = _process_route(
            route,
            base_argv=base_argv,
            config=config,
            config_path=config_path,
            force_rebuild=force_rebuild,
        )
        processed += route_processed
        recorded += route_recorded
    print(
        f"Level 0 incremental processing complete: {processed:,} new/replaced "
        f"file(s) processed across {len(routes):,} route(s); "
        f"{recorded:,} file(s) recorded."
    )


if __name__ == "__main__":
    operational_main()
