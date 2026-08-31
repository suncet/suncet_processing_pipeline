"""Bootstrap a SunCET development checkout with Mamba.

This helper does not edit shell profiles or guess private paths. It creates or
updates the named Mamba environment, installs the checkout in editable mode, and
initializes the public data tree using explicit absolute paths. It validates
reviewed metadata already delivered through Dropbox/rclone and never downloads
the mutable live Google Sheet.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import setup_minimum_required_folders_files


def _absolute_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"expected an absolute path, got {value!r}")
    return path.resolve(strict=False)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create/update the SunCET Mamba environment and data tree."
    )
    parser.add_argument("--data-root", required=True, type=_absolute_path)
    parser.add_argument("--ctdb-root", required=True, type=_absolute_path)
    parser.add_argument("--environment", default="suncet")
    parser.add_argument(
        "--definition",
        type=Path,
        default=Path(__file__).resolve().parent / "environment.yml",
    )
    parser.add_argument(
        "--skip-environment",
        action="store_true",
        help="Only initialize the data tree (useful after an environment exists).",
    )
    setup_minimum_required_folders_files.add_metadata_policy_arguments(parser)
    return parser


def _environment_exists(mamba: str, environment: str) -> bool:
    result = subprocess.run(
        [mamba, "env", "list", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    return any(
        Path(path).name == environment for path in json.loads(result.stdout)["envs"]
    )


def bootstrap(argv: list[str] | None = None) -> Path:
    args = get_parser().parse_args(argv)
    if not args.ctdb_root.is_dir():
        raise SystemExit(f"Private CTDB directory does not exist: {args.ctdb_root}")
    args.data_root.mkdir(parents=True, exist_ok=True)
    if (
        args.ctdb_root == args.data_root
        or args.ctdb_root.is_relative_to(args.data_root)
        or args.data_root.is_relative_to(args.ctdb_root)
    ):
        raise SystemExit("suncet_data and suncet_ctdb must not overlap")

    repository = Path(__file__).resolve().parent
    if not args.skip_environment:
        mamba = shutil.which("mamba")
        if mamba is None:
            raise SystemExit(
                "Mamba was not found. Install Miniforge, then rerun this command."
            )
        if _environment_exists(mamba, args.environment):
            environment_command = [
                mamba,
                "env",
                "update",
                "--name",
                args.environment,
                "--file",
                str(args.definition),
                "--prune",
            ]
        else:
            environment_command = [
                mamba,
                "env",
                "create",
                "--name",
                args.environment,
                "--file",
                str(args.definition),
            ]
        subprocess.run(environment_command, check=True, cwd=repository)
        subprocess.run(
            [
                mamba,
                "run",
                "--name",
                args.environment,
                "python",
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--editable",
                str(repository),
            ],
            check=True,
            cwd=repository,
        )

    os.environ["suncet_data"] = str(args.data_root)
    os.environ["suncet_ctdb"] = str(args.ctdb_root)
    initializer_args = (
        ["--allow-missing-metadata"] if args.allow_missing_metadata else []
    )
    data_root = setup_minimum_required_folders_files.run(initializer_args)

    print("\nAdd these lines to the appropriate shell profile on this host:")
    print(f"export suncet_data={str(args.data_root)!r}")
    print(f"export suncet_ctdb={str(args.ctdb_root)!r}")
    print(f"Activate with: mamba activate {args.environment}")
    print(f"Validate with: mamba run -n {args.environment} python -m pytest")
    return data_root


if __name__ == "__main__":
    bootstrap()
