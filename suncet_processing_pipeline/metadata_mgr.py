import pandas as pd
from pathlib import Path


class MetadataManager:
    """Class for interacting with SunCET Metadata files.

    This expect the metadata is downloaded into the run directory. To do that,
    see: setup_minimum_required_folders_files.py
    """
    def __init__(self, run_dir):
        """Initialize a metadata manager from a run directory, which
        is expected to have the required files.

        Args
           run_dir: Path to run directory
        """
        # Set paths and check they exist
        self._metadata_path = Path(run_dir) / 'suncet_metadata_definition.csv'
        self._metadata_ver_path = Path(run_dir) / 'suncet_metadata_definition_version.csv'

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
        self._metadata_df = pd.read_csv(self._metadata_path)
        self._metadata_df = _clean_metadata_comments(self._metadata_df)
        
        # Load metadata version (just read string from text file)
        with open(self._metadata_ver_path) as fh:
            self._metadata_ver = fh.read().strip()
            
        print(f'Found metadata version "{self._metadata_ver}"')

        # Convert metadata df to dictionary mapping internal name to dictionary
        # of columns -> values
        self._metadata_dict = _get_metadata_dict(self._metadata_df)

    def get_netcdf4_variable_name(self, internal_name):
        """Get name of variable for writing to a NetCDF4 file

        Args
          internal_name: Internal name of variable (within code)
        Returns
          what that internal name should be called in a NetCDF4 file
        """
        # Ensure variable is in the metadata dictionary
        if internal_name not in self._metadata_dict:
            raise RuntimeError(
                f"Could not find metadata for variable with "
                f"internal name '{internal_name}'"
            )

        # Get the variable name, raising Exception if its not filled out in the
        # table
        var_name = self._metadata_dict[internal_name]['netCDF variable name']

        if not var_name:
            raise RuntimeError(
                f"Needed NetCDF variable name for internal name "
                f"{internal_name}\", but missing"
            )

        # Return good result
        return var_name

    def get_netcdf4_dimension_names(self, internal_name):
        """Get tuple of dimension names for the given variable.
        
        Args
          internal_name: Internal name of variable (within code)
        Returns
          tuple of dimension names
        """
        # Ensure variable is in the metadata dictionary
        if internal_name not in self._metadata_dict:
            raise RuntimeError(
                f"Could not find metadata for variable with internal name "
                f"'{internal_name}'."
            )
        # Load variable dict and return subset of keys that are relevant        
        var_dict = self._metadata_dict[internal_name]
        dim_csv = var_dict['netCDF dimensions']
        
        if dim_csv:        
            return tuple(dim_csv.split(','))
        else:
            return tuple()  # empty tuple
    
    
    def get_netcdf4_attrs(self, internal_name):
        """Get dictionary of static NetCDF4 attributes for a given variable.

        Args
          internal_name: Internal name of variable (within code)
        Returns
          dictionary of attribute keys to values
        """
        # Ensure variable is in the metadata dictionary
        if internal_name not in self._metadata_dict:
            raise RuntimeError(
                f"Could not find metadata for variable with internal name "
                f"'{internal_name}'."
            )
        # Load variable dict and return subset of keys that are relevant        
        var_dict = self._metadata_dict[internal_name]

        return {
            "units": var_dict["units (human)"]    
        }
        
        
def _get_metadata_dict(metadata_df):
    """Convert metadata dataframe to dictinoary mapping internal name
    to dictionary of cols to values.

    Args
      metadata_df: Metadata dictionary as loaded from flie with comments
        cleaned
    Returns
       dictionary mapping internal names to dictionaries holding the
       row information.
    """
    metadata_dict = {}
    
    for _, row in metadata_df.iterrows():
        cur_dict = {col: row[col] for col in metadata_df.columns}
        cur_key = row['Internal Variable Name']

        metadata_dict[cur_key] = cur_dict

    return metadata_dict
        

def _clean_metadata_comments(metadata_df):
    """Remove comment rows from the metadata Data Frame.
    
    A command has the work "COMMENT" in the first column

    Args
      dataframe as loaded directly from CSV file
    Returns
      dataframe with comment row dropped
    """
    collected_rows = []
    first_row = metadata_df.columns[0]
    
    for _, row in metadata_df.iterrows():
        if 'COMMENT' not in row[first_row].upper():
            collected_rows.append(row)
        
    return pd.DataFrame(collected_rows)
        
