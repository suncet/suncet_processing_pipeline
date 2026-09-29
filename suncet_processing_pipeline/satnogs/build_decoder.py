"""Rebuild or check all public decoder artifacts using Kaitai compiler 0.11."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

from .kaitai_generator import generate_kaitai
from .synthetic_fixture import build_synthetic_fixture


COMPILER_VERSION = "0.11"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="kaitai-struct-compiler")
    parser.add_argument("--check", action="store_true", help="Fail on stale artifacts without writing")
    args = parser.parse_args()

    version = subprocess.run(
        [args.compiler, "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if version != f"kaitai-struct-compiler {COMPILER_VERSION}":
        parser.error(f"Expected compiler {COMPILER_VERSION}, got {version!r}")

    ksy = generate_kaitai()
    with tempfile.TemporaryDirectory(prefix="suncet-kaitai-") as directory:
        temporary = Path(directory)
        source = temporary / "suncet_apid1.ksy"
        source.write_text(ksy, encoding="utf-8")
        subprocess.run(
            [args.compiler, "--target", "python", "--outdir", directory, str(source)],
            check=True,
        )
        compiled = (temporary / "suncet_apid1.py").read_text(encoding="utf-8")

    # The compiler emits spaces on blank docstring lines; normalize only whitespace.
    compiled = "\n".join(line.rstrip() for line in compiled.splitlines()).rstrip() + "\n"
    packet, expected = build_synthetic_fixture()
    artifacts = {
        "suncet_apid1.ksy": ksy,
        "generated_suncet_apid1.py": compiled,
        "test_data/suncet_apid1_synthetic_252.hex": packet.hex() + "\n",
        "test_data/suncet_apid1_synthetic_252_expected.json": (
            json.dumps(expected, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ),
    }
    root = Path(__file__).parent
    stale = []
    for name, contents in artifacts.items():
        destination = root / name
        if args.check:
            if not destination.is_file() or destination.read_text(encoding="utf-8") != contents:
                stale.append(name)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(contents, encoding="utf-8")
    if stale:
        parser.exit(1, "Stale public decoder artifacts: " + ", ".join(stale) + "\n")
    print(f"{'Verified' if args.check else 'Wrote'} {len(artifacts)} public decoder artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
