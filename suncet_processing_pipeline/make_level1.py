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
from sunkit_image.utils.noise_estimation import noise_estimate
from suncet_processing_pipeline import config_parser


class Level1:
    def __init__(self, config):
        self.config = config
        self.metadata = self.__load_metadata_from_level0_5()
        self.cosmic_rays = np.array([])

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

        dark_corrected = self.__dark_correction(level05_map, os.path.join(self.config.calibration_path,
                                                                          self.config.dark_filename))
        flat_corrected = self.__flat_correction(dark_corrected, os.path.join(self.config.calibration_path,
                                                                             self.config.flat_filename))
        badpixel_corrected = self.__badpixel_removal(flat_corrected, os.path.join(self.config.calibration_path,
                                                                                  self.config.badpix_filename))
        cosmic_ray_corrected = self.__cosmic_ray_removal(badpixel_corrected)

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
            row_indices, col_indices = np.where(outlier_mask)
            removed_values = input_map.data[row_indices, col_indices]

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

            corrected_values = cosmic_ray_corrected_data[row_indices, col_indices]
            index_1d_ravel = np.ravel_multi_index((row_indices, col_indices), input_map.data.shape)

            # Set cosmic ray output file (1d location, removed spike value, interpolated spike value)
            self.cosmic_rays = np.column_stack((index_1d_ravel, removed_values, corrected_values))
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

        # Update metadata dictionary
        self.metadata.coord_sys_rotation = k * 90.
        self.metadata.wcs_rot_pc11, self.metadata.wcs_rot_pc12, self.metadata.wcs_rot_pc21, self.metadata.wcs_rot_pc22 = \
            (utilities.CROTA_2_WCSrotation_matrix(k * 90.))

        return np.rot90(data, k)

    def __create_exposure_time_mask(self, image_data): #TODO check composite image dimensions expected
        """
        Create an exposure time mask matching the shape of image_data.

        Parameters:
        -----------
        image_data : np.ndarray
            Composite image data, shape can be (n_sections, height, width) or (height, width)

        Returns:
        --------
        np.ndarray
            Exposure time mask with the same shape as image_data.
        """
        telapse = self.metadata.TELAPSE

        if image_data.ndim == 3:
            telapse = np.asarray(telapse)
            if telapse.size != image_data.shape[0]:
                raise ValueError("Length of TELAPSE does not match number of image sections.")
            # Broadcast exposure times across height and width
            exposure_mask = telapse[:, None, None] * np.ones_like(image_data)
        elif image_data.ndim == 2:
            # Single-section image: broadcast scalar or single-element array
            exposure_mask = np.full_like(image_data, telapse, dtype=float)
        else:
            raise ValueError("Unsupported image_data shape: must be 2D or 3D array.")

        return exposure_mask

    def __convert_to_DN(self, image_data):
        """
        Convert image data to DN by dividing each pixel by its exposure time.

        Parameters:
        -----------
        image_data : np.ndarray
            Composite image data, expected to be 2D or 3D

        Returns:
        --------
        np.ndarray
            Image data converted to DN.
        """
        exposure_mask = self.__create_exposure_time_mask(image_data)
        return image_data / exposure_mask

def __compute_1AU_scaling_factor(self): #TODO decide where this belongs in the meta data
    """
    Compute a scaling factor to adjust image intensity to 1 AU distance.

    Returns:
    --------
    float
        Multiplicative scaling factor to adjust observed intensity to what it would be at 1 AU.
    """
    AU_METERS = 1.496e11  # 1 AU in meters
    dsun_obs = self.metadata.DSUN_OBS

    if dsun_obs is None or dsun_obs <= 0:
        raise ValueError("Invalid DSUN_OBS value: must be a positive number in meters.")

    # Inverse-square law: intensity ∝ 1 / distance^2 → scale by (AU / dsun_obs)^2
    scaling_factor = (AU_METERS / dsun_obs) ** 2
    return scaling_factor


def parameterize_noise(self, image_data, n_regions=9, region_size=64): #TODO test for desired outputs
    """
    Estimate image noise sigma in representative regions and store results in metadata.

    Parameters:
    -----------
    image_data : np.ndarray
        2D or 3D array representing image(s).
    n_regions : int
        Total number of regions to sample (must be a perfect square for grid).
    region_size : int
        Size of square region in pixels for each sample.

    Stores:
    -------
    self.metadata.noise_map : dict
        Dictionary with region locations and corresponding noise sigma estimates.
    """
    if image_data.ndim == 3:
        # Average over sections to get a representative 2D image
        image = np.mean(image_data, axis=0)
    elif image_data.ndim == 2:
        image = image_data
    else:
        raise ValueError("Unsupported image_data shape: must be 2D or 3D.")

    height, width = image.shape
    noise_map = {}
    grid_dim = int(np.sqrt(n_regions))
    if grid_dim ** 2 != n_regions:
        raise ValueError("n_regions must be a perfect square (e.g., 9, 16, 25).")

    step_y = height // grid_dim
    step_x = width // grid_dim

    for i in range(grid_dim):
        for j in range(grid_dim):
            y0 = i * step_y + (step_y - region_size) // 2
            x0 = j * step_x + (step_x - region_size) // 2

            # Ensure boundaries are within the image
            y0 = max(0, min(y0, height - region_size))
            x0 = max(0, min(x0, width - region_size))

            region = image[y0:y0 + region_size, x0:x0 + region_size]
            sigma = noise_estimate(region)

            noise_map[(i, j)] = {
                "center": (y0 + region_size // 2, x0 + region_size // 2),
                "sigma": float(sigma)
            }

    self.metadata.noise_map = noise_map



if __name__ == "__main__":
    level1 = Level1()
    level1.run()

