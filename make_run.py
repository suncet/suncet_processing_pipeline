"""Script to start a new run below ``$suncet_data/processing_runs``."""

import argparse
from pathlib import Path
import shutil
import sys

from termcolor import cprint

from suncet_processing_pipeline.data_paths import processing_run_path
from suncet_processing_pipeline.metadata_snapshots import snapshot_metadata_for_run


def main():
    """Main function of the program."""
    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Decide on run path and check if it already exists
    run_dir = processing_run_path(args.run_name)

    if run_dir.exists():
        cprint(f"Error: {run_dir} already exists", "red")
        sys.exit(1)
        
    # Make the run directory
    run_dir.mkdir(parents=True)
    cprint(f"Directory {run_dir} created", "green")

    # Make the subdirectories
    subdirs = ['input', 'level1', 'level2', 'level3']
    
    for subdir in subdirs:
        subdir_path = run_dir / subdir
        subdir_path.mkdir()
        print(f"  Sub-directory ", end="")
        cprint(subdir_path, "yellow", end=" ")
        print("created")

    # Copy the default config.ini
    config_path = run_dir / 'config.ini'
    default_config = (
        Path(__file__).resolve().parent
        / 'suncet_processing_pipeline'
        / 'config_files'
        / 'config_default.ini'
    )

    shutil.copy(default_config, config_path)
    print(f"Copied ", end="")
    cprint(config_path, "yellow")

    metadata_manifest = snapshot_metadata_for_run(run_dir)
    print(
        "Snapshotted metadata definition "
        f"v{metadata_manifest['metadata_version']} into this run"
    )

    # Print final message stating success
    print("Run creation completed successfully")
    
    
def get_parser():
    """Get command line ArgumentParser object with options defined.
        
    Returns
    -------
    parser : argparse.ArgumentParser
       object which can be used to parse command line objects
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--run-name', type=str, required=True,
                        help='String name of the run')

    return parser


if __name__ == '__main__':
    main()
