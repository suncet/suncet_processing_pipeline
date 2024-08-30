"""
This is the code to make the Level 3 data product. 
"""
import argparse



class Level3:
    """Class for applying the Level2 -> Level3 processing stage.

    The key method is `run()`, which acts lik a main() method for
    this stage.
    """
    def __init__(self, config):
        """
        Parameters
        ----------
        level2_data : dict, str -> array
            Level 2 data, mapping internal variable names to their values 
            (generally numpy arrays)
        config : config_parser.Config
            SunCET Data Processing Pipeline configration object
        """
        self.config = config

    def run(self):
        """Main method to process the level2 -> level3 stage."""
        # Parse command line arguments
        parser = self._get_parser()
        args = parser.parse_args()

    def _get_parser(self):
        """Get command line ArgumentParser object with options defined.
        
        Returns
        -------
        parser : argparse.ArgumentParser
           object which can be used to parse command line objects
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('-v', '--verbose', help='Print more debugging output')

        return parser


def final_shdr_compositing_fix(level2_data, config):
    """Fix any lingaring SHDR Compositing Issues.

    Parameters
    ----------
    level2_data : dict, str -> array
      Level 2 data, mapping internal variable names to their values 
       (generally numpy arrays)
    config : config_parser.Config
       SunCET Data Processing Pipeline configration object

    Returns
    -------
    level2_data_fixed : dict, str -> array
       Copy of level2 data with the fix applied.
    """
    raise NotImplementedError()
    

if __name__ == '__main__':
    level3 = Level3()
    level3.run()
