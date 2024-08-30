"""Script to start a new run in the processing_runs/ directory."""

import argparse
import os
import shutil
import sys

from termcolor import cprint


def main():
    """Main function of the program."""
    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Decide on run path and check if it already exists
    run_dir = os.path.join('processing_runs', args.run_name)

    if os.path.exists(run_dir):
        cprint(f"Error: {run_dir} already exists", "red")
        sys.exit(1)
        
    # Make the run directory
    os.mkdir(run_dir)
    cprint(f"Directory {run_dir} created", "green")

    # Make the subdirectories
    subdirs = ['input', 'level1', 'level2', 'level3']
    
    for subdir in subdirs:
        subdir_path = os.path.join(run_dir, subdir)
        os.mkdir(subdir_path)
        cprint(f"  Sub-directory {subdir_path} created", "green")

    # Copy the default config.ini
    config_path = os.path.join(run_dir, 'config.ini')
    
    shutil.copy('suncet_processing_pipeline/config_files/config_default.ini', config_path)
    cprint(f"Copied {config_path}", "green")

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
