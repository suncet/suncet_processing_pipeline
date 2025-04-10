"""Metadata Management Classes

Usage:

   metadata = metadata_mgr.FitsMetadataManager(self.run_dir)
   metadata.load_from_dict({
      ...
   })
        
   metadata.generate_fits_header(fits_file)
"""
import pandas as pd
from pathlib import Path


# Expected name of FITS metadata definitions file in the run directory
FITS_METADATA_DEFINITIONS_FILENAME = 'suncet_metadata_definition_fits.csv'

# Expected name of the meta version text file in the run directory
FITS_METADATA_VERSION_FILENAME = 'suncet_metadata_definition_version.txt'


class FitsMetadataManager:
    """Class for interacting with SunCET FITS Metadata files.

    This expect the metadata is downloaded into the run directory. To do that,
    see: setup_minimum_required_folders_files.py
    """
    def __init__(self, run_dir):
        """Initialize a metadata manager from a run directory, which
        is expected to have the required files.

        Args
           run_dir: Path to run directory
        """
        # Convert run directory to a Path object
        run_dir = Path(run_dir)
        
        # Set paths and check they exist
        self._metadata_path = run_dir / FITS_METADATA_DEFINITIONS_FILENAME
        self._metadata_ver_path = run_dir / FITS_METADATA_VERSION_FILENAME

        if not self._metadata_path.exists():
            raise FileNotFoundError(
                f"Error: could not find metadata at {self._metadata_path}"
            )
            
        if not self._metadata_ver_path.exists():
            raise FileNotFoundError(
                f"Error: could not find metadata version at {self._metadata_path}"
            )

        # Load metadata CSV using Pandas
        print(f'Reading metadata from {self._metadata_path}')
        self._df_metadata = pd.read_csv(self._metadata_path)

        # Group internal names by the comment block they are in. This
        # is a dictionary that maps group name to a list of internal
        # names
        self._metadata_groups = _get_metadata_groups(self._df_metadata)

        # Remove comments from datafarme after this
        self._df_metadata = _clean_metadata_comments(self._df_metadata)
        
        # Load metadata version (just read string from text file)
        with open(self._metadata_ver_path) as fh:
            self._metadata_ver = fh.read().strip()
            
        print(f'Found metadata version "{self._metadata_ver}"')

        # Convert metadata df to dictionary mapping internal name to dictionary
        # of spreadsheet's columns to cell contents
        self._metadata_dict = _get_metadata_dict(self._df_metadata)
        
        # Define variable that will be used to store values. Maps internal name
        # to values
        self._metadata_values = {}        
        
    def load_from_dict(self, metadata_values):
        """Load metadata values from a dictionary. 

        This can be called subsequent times to incrementally add metadata
        values.

        Args
           metadata_values: keys should be metadata internal names
        """        
        self._metadata_values.update(metadata_values)

    def generate_fits_header(self, fits_file):
        """Add a FITS header to an open fits file using metadata values
        which have been supplied.

        Args
           fits_file: object returned by fits.open()
        See also:
           load_from_dict() to add metadata values
        """
        # Build list of things to write, organized by the groups they are
        # in with care to preserve their order
        vars_with_values = set(self._metadata_values.keys())
        counter = 0
        to_write = []
        
        for group_name, group_variables in self._metadata_groups.items():
            # Do nothing if no variables with values in this group
            if len(set(group_variables) & vars_with_values) == 0:
                continue

            # Add group line for this block
            to_write.append((counter, 'COMMENT', group_name.center(72, '-')))
            counter += 1

            # Add variables under this group
            for group_var in group_variables:
                if group_var in vars_with_values:
                    name = self._metadata_dict[group_var]['FITS variable name']
                    description = self._metadata_dict[group_var]['Description']

                    value = self._metadata_values[group_var]
                    to_write.append((counter, name, value, description))
                    counter += 1

        # Write to_write items to fits header
        header = fits_file[0].header
        
        for index, *args in to_write:
            header.insert(index, tuple(args))
        
        
def _get_metadata_dict(df_metadata):
    """Convert metadata dataframe to dictinoary mapping internal name
    to dictionary of cols to values.

    Args
      df_metadata: Metadata dictionary as loaded from flie with comments
        cleaned
    Returns
       dictionary mapping internal names to dictionaries holding the
       row information.
    """
    metadata_dict = {}
    
    for _, row in df_metadata.iterrows():
        cur_dict = {col: row[col] for col in df_metadata.columns}
        cur_key = row['Internal Variable Name']
        metadata_dict[cur_key] = cur_dict

    return metadata_dict
        

def _clean_metadata_comments(df_metadata):
    """Remove comment rows from the metadata Data Frame.
    
    A command has the work "COMMENT" in the first column

    Args
      dataframe as loaded directly from CSV file
    Returns
      dataframe with comment row dropped
    """
    collected_rows = []
    first_row = df_metadata.columns[0]
    
    for _, row in df_metadata.iterrows():
        if 'COMMENT' not in str(row[first_row]).upper():
            collected_rows.append(row)
        
    return pd.DataFrame(collected_rows)
        

def _get_metadata_groups(df_metadata):
    """Get dictionary that maps group names (COMMENT blocks) to
    list of internal variable names in that group.
    
    Args
      dataframe as loaded directly from CSV file
    Returns
       dictionary mapping group names to list of internal
       variable names
    """
    row_groups = {}
    group_name = None
    first_row = df_metadata.columns[0]

    for _, row in df_metadata.iterrows():
        if 'COMMENT' in str(row[first_row]).upper():
            group_name = (
                row[first_row]
                .replace('COMMENT', '')
                .replace('-', '')
                .strip()
            )
            row_groups[group_name] = []            
        elif group_name:
            row_groups[group_name].append(row['Internal Variable Name'])

    return row_groups
