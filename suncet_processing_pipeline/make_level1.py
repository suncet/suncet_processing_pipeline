"""
This is the code to make the Level 1 data product. 
"""
import os
from glob import glob
import copy
from pathlib import Path
import astropy.units as u
from astropy.io import fits
import numpy as np
import sunpy.map
from scipy import ndimage, interpolate
from pykdtree.kdtree import KDTree
from suncet_processing_pipeline import suncet_utility as utilities
from suncet_processing_pipeline import config_parser


class Level1:
    def __init__(self, config):
        self.config = config
        # self.metadata = self.__load_metadata_from_level0_5()
        self.cosmic_rays = np.array()

    def __load_metadata_from_level0_5(self):
         pass

    def make(self, level0_5_to_process=None): 
        if level0_5_to_process is None: 
        #    raise ValueError('Need to provide either path to files (string) or list of filenames that you want to process.') # FIXME: uncomment this once we don't need the hack to get a synthetic image
            pass
        if isinstance(level0_5_to_process, list):
            filenames = level0_5_to_process
        elif isinstance(level0_5_to_process, str):
            filenames = glob(os.path.join(level0_5_to_process, '*.fits'))
        else: 
            raise TypeError('Need to provide either path to files (string) or list of filenames that you want to process.')
        
        filenames = os.getenv('suncet_data') + '/synthetic/level0_raw/fits/config_default_OBS_2023-02-14T17:00:00.000_300.fits' # Hack to get synthetic image 
        print("make_level1 is going to process the following input files:")
        for file in filenames:
            print(f"- {file}")

        level0_5 = self.__load_level0_5(filenames)
        
        meta_filename = self.__make_metadata_filename(filenames, self.config.version)
        #self.save_metadata(filename=meta_filename)

        pass

    def __load_level0_5(self, filenames):
        # map_list = []
        # for file in filenames:
        #     map_list.append(sunpy.map.Map(file))

        # return sunpy.map.MapSequence(map_list)
        level05_map = sunpy.map.Map(filenames)
        return level05_map

    def __make_metadata_filename(self, filename, version):
        filename_with_extension = os.path.basename(filename)
        base, extension = os.path.splitext(filename_with_extension)
        return f"{base}{'_metadata_v'}{version}{'.csv'}"

    def run(self):
        pass

    def apply_basic_corrections(self, filename, dark_filename, flat_filename, badpixel_filename):
        # Expects a map of level 0.5
        # Walks through a dark correction, flat field correction, bad pixel removal, and cosmic ray removal
        # Returns a sunpy map

        level05_map = self.__load_level0_5(filename)

        dark_corrected = self.__dark_correction(self, level05_map,
                                                os.path.join(self.config.calibration_path, self.config.dark_filename))
        flat_corrected = self.__flat_correction(self, dark_corrected,
                                                os.path.join(self.config.calibration_path, self.config.flat_filename))
        badpixel_corrected = self.__badpixel_removal(self, flat_corrected,
                                                     os.path.join(self.config.calibration_path, self.config.badpix_filename))
        outlier_mask = self.detect_outliers(badpixel_corrected.data)
        cosmic_ray_corrected = self.__cosmic_ray_removal(self, badpixel_corrected, outlier_mask)

        basic_map = cosmic_ray_corrected

        return basic_map

    def __dark_correction(self, input_map, dark_image_path):
        # Read the target image and dark frame and extract data
        with fits.open(dark_image_path) as dark_hdul:

            dark_data = dark_hdul.data[0].data

            # Subtract dark frame from the target image to correct for dark bias
            corrected_data = input_map.data - dark_data

            # Update the data in the target map with corrected data
            dark_corrected_map = input_map
            dark_corrected_map.data[:] = corrected_data[:]

        return dark_corrected_map

    def __flat_correction(self, input_map, flat_image_path):
        # Read the target image and flat frame and extract data
        with fits.open(flat_image_path) as flat_hdul:

            flat_data = flat_hdul.data[0].data

            # Divide flat frame from the target image to correct for flat bias
            corrected_data = input_map.data / flat_data

            # Update the data in the target map with corrected data
            flat_corrected_map = input_map
            flat_corrected_map.data[:] = corrected_data[:]

        return flat_corrected_map

    def __badpixel_removal(self, input_map, badpixel_path):
        # Read the target image and bad pixel frame and extract data
        with fits.open(badpixel_path) as bad_pixel_hdul:
            bad_pixel_mask = bad_pixel_hdul[0].data

            # Create a new array with bad pixels replaced with NaN
            corrected_data = np.where(bad_pixel_mask == 1, np.nan, input_map.data)

            # Update the data in the target map with corrected data
            badpixel_corrected_map = input_map
            badpixel_corrected_map[:] = corrected_data[:]

        return badpixel_corrected_map

    def __cosmic_ray_removal(self, input_map):
        """
        Interpolates over bad pixels in the input image using linear interpolation if the config file has
        cosmic_ray_removal set to True.

        Parameters:
            input_map (sunpy.map): Input 2D map (image) containing pixel values.

        Returns:
            sunpy.map: Interpolated sunpy map with bad pixels replaced by interpolated values.
        """
        if self.config.cosmic_ray_removal:

            outlier_mask = utilities.detect_outliers(input_map.data, 20)

            # Create coordinate grid for good pixels
            y, x = np.indices(input_map.data.shape)
            good_pixel_coords = np.column_stack((y[~outlier_mask], x[~outlier_mask]))

            # Values of good pixels
            good_pixel_values = input_map.data[~outlier_mask]

            # Coordinates of bad pixels
            bad_pixel_coords = np.column_stack((y[outlier_mask], x[outlier_mask]))

            # Build the kdTree
            tree = KDTree(good_pixel_coords)

            # Query the nearest neighbors for the bad pixel coordinates
            distances, indices = tree.query(bad_pixel_coords, k=4)  # k=4 nearest neighbors

            # Retrieve the coordinates and values of the 4 nearest neighbors
            neighbor_coords = good_pixel_coords[indices]
            neighbor_values = good_pixel_values[indices]

            # Compute bilinear weights based on the inverse of the distances
            weights = 1 / (distances + 1e-12)  # Avoid division by zero
            weights /= np.sum(weights, axis=1, keepdims=True)

            # Compute the linear interpolated values as a weighted sums
            interpolated_values = np.sum(weights * neighbor_values, axis=1)

            # A slow cubic interpolation at the `bad_pixel_coords` points
            # interpolated_values = interpolate.griddata(good_pixel_coords, good_pixel_values, bad_pixel_coords, method='cubic')

            # Replace bad pixel values with interpolated values
            cosmic_ray_corrected_data = copy.deepcopy(input_map.data)
            cosmic_ray_corrected_data[bad_pixel_coords[:, 0], bad_pixel_coords[:, 1]] = interpolated_values
        else:
            cosmic_ray_corrected_data = input_map.data.copy()

        return sunpy.map.Map(cosmic_ray_corrected_data, input_map.meta.copy())

    def __coarse_rotate(self, data, telemetry):
        """
        A rotation by a factor of 90 degrees to ensure solar north is in the top quadrant.
        Updates the rotation metadata

        Parameters:
            data (numpy.ndarray): Input 2D array (image) containing pixel values.
            telemetry (dict): a dictionary of single point values from SunCET

       Returns:
           numpy.ndarray: 2D array (image) containing pixel values rotated to keep solar north in the top quadrant
        """

        angle_deg = utilities.detector_angle(telemetry)

        # Determining the number of 90 deg rotations to keep solar north approximately on the top of the matrix
        k = np.round(angle_deg / 90)

        # TODO: update self.metadata rotation to indicate the number of rotations

        return np.rot90(data, k)


if __name__ == "__main__":
    level1 = Level1()
    level1.run()

