"""
This is the code to make the Level 2 data product.

Level 2 processing applies deconvolution to Level 1 data to remove
diffraction and scatter effects from the PSF.
"""
import argparse
from pathlib import Path
from glob import glob
import os

from astropy.io import fits
import numpy as np
from termcolor import cprint

from suncet_processing_pipeline import config_parser
from suncet_processing_pipeline import suncet_deconv


class Level2:
    """Class for applying the Level1 -> Level2 processing stage (deconvolution)."""
    
    def __init__(self, config=None, 
                 diffraction_psf_file=None, scatter_psf_file=None,
                 spec_file=None, resp_file=None, correction_factor=0.4):
        """
        Args:
            config: config_parser.Config, SunCET Data Processing Pipeline configuration object
            diffraction_psf_file: Path to diffraction PSF file
            scatter_psf_file: Path to scatter PSF file  
            spec_file: Path to spectral file
            resp_file: Path to response file
            correction_factor: Deconvolution correction factor (default 0.4)
        """
        # Load config if not provided
        if config is None:
            default_config_path = Path(__file__).parent / 'config_files' / 'config_default.ini'
            self.config = config_parser.Config(str(default_config_path))
        else:
            self.config = config
            
        self.diffraction_psf_file = diffraction_psf_file
        self.scatter_psf_file = scatter_psf_file
        self.spec_file = spec_file
        self.resp_file = resp_file
        self.correction_factor = correction_factor
    
    def run(self, input_path, output_path=None):
        """
        Process all FITS files in input_path and apply deconvolution.
        
        Args:
            input_path: Path to directory containing Level 1 FITS files or path to single file
            output_path: Path to directory where Level 2 files will be saved (optional)
        """
        # Determine input files
        input_path = Path(input_path)
        if input_path.is_dir():
            fits_files = sorted(glob(str(input_path / '*.fits')))
        elif input_path.is_file():
            fits_files = [str(input_path)]
        else:
            raise ValueError(f"Input path does not exist: {input_path}")
        
        if not fits_files:
            cprint(f"No FITS files found in {input_path}", 'yellow')
            return
        
        # Determine output directory
        if output_path is None:
            suncet_data = os.getenv('suncet_data')
            if suncet_data is None:
                raise ValueError("Environment variable 'suncet_data' is not set")
            
            # Check if we're working with synthetic data
            if 'synthetic' in str(input_path):
                output_path = Path(suncet_data) / 'synthetic' / 'level2'
            else:
                output_path = Path(suncet_data) / 'level2'
        else:
            output_path = Path(output_path)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        cprint(f"Processing {len(fits_files)} FITS file(s) from {input_path}", 'green')
        cprint(f"Output directory: {output_path}", 'green')
        print()
        
        # Process each file
        for fits_file in fits_files:
            self._process_single_file(fits_file, output_path)
    
    def _process_single_file(self, input_file, output_dir):
        """
        Process a single FITS file through deconvolution.
        
        Args:
            input_file: Path to input FITS file
            output_dir: Directory where output will be saved
        """
        input_file = Path(input_file)
        print(f"Processing: {input_file.name}")
        
        # Load Level 1 data (or apply crude calibration if Level 0)
        l1_data = suncet_deconv._crude_calibration_level0(str(input_file))
        
        # Apply deconvolution
        decon_data = suncet_deconv.apply_deconv(
            l1_data,
            self.diffraction_psf_file,
            self.scatter_psf_file,
            self.resp_file,
            self.spec_file,
            correction_factor=self.correction_factor,
        )
        
        # Create output filename with version
        base_name = input_file.stem  # filename without extension
        output_file = output_dir / f"{base_name}_level2_v{self.config.version}.fits"
        
        # Save to FITS file
        self._save_fits(decon_data, input_file, output_file)
        
        cprint(f"  Saved to: {output_file.name}", 'green')
        print()
    
    def _save_fits(self, data, input_file, output_file):
        """
        Save deconvolved data to FITS file, preserving original header.
        
        Args:
            data: Deconvolved image data array
            input_file: Original input file (to copy header from)
            output_file: Output file path
        """
        # Read original header
        with fits.open(input_file) as hdul:
            header = hdul[0].header.copy()
        
        # Update header to indicate Level 2 processing
        header['LEVEL'] = 2
        header['HISTORY'] = 'Applied deconvolution for diffraction and scatter'
        header['DECONV'] = True
        header['DECONVCF'] = (self.correction_factor, 'Deconvolution correction factor')
        
        # Create HDU and write
        hdu = fits.PrimaryHDU(data=data, header=header)
        hdu.writeto(output_file, overwrite=True)


def _get_parser():
    """Get command line ArgumentParser object with options defined.
    
    Returns:
        ArgumentParser object which can be used to parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Apply deconvolution to SunCET Level 1 data to produce Level 2'
    )
    parser.add_argument('-i', '--input-path', type=str, required=True,
                        help='Path to directory containing Level 1 FITS files or single file')
    parser.add_argument('-o', '--output-path', type=str, default=None,
                        help='Path to output directory (default: $suncet_data/level2/ or $suncet_data/synthetic/level2/)')
    parser.add_argument('-c', '--config-file', type=str, default=None,
                        help='Path to config file (default: config_files/config_default.ini)')
    parser.add_argument('--diffraction-psf-file', type=str, required=True,
                        help='Path to diffraction PSF file')
    parser.add_argument('--scatter-psf-file', type=str, required=True,
                        help='Path to scatter PSF file')
    parser.add_argument('--spec-file', type=str, required=True,
                        help='Path to spectral file')
    parser.add_argument('--resp-file', type=str, required=True,
                        help='Path to response file')
    parser.add_argument('--correction-factor', type=float, default=0.4,
                        help='Deconvolution correction factor (default: 0.4)')
    
    return parser


def main():
    """Main method when running this script directly."""
    args = _get_parser().parse_args()
    
    # Load config if provided
    config = None
    if args.config_file:
        config = config_parser.Config(args.config_file)
    
    # Create Level2 instance
    level2 = Level2(
        config=config,
        diffraction_psf_file=args.diffraction_psf_file,
        scatter_psf_file=args.scatter_psf_file,
        spec_file=args.spec_file,
        resp_file=args.resp_file,
        correction_factor=args.correction_factor,
    )
    
    # Run processing
    level2.run(args.input_path, args.output_path)


if __name__ == '__main__':
    main()

