"""
Helper functions for the processing pipeline
"""
import astropy.units as u
from astropy.coordinates import ICRS, GCRS, Angle
from astropy.time import Time
from astropy.io import fits
from astropy.io.fits.verify import VerifyError
import sunpy.map
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation as R


def detect_outliers(data, threshold=2):
    """
    Detects pixels in the input 2D array that deviate significantly from their 8 nearest neighbors.

    Parameters:
        data (numpy.ndarray): Input 2D array (image) containing pixel values.
        threshold (float): Relative Threshold for deviation from neighbors. Pixels deviating
                          more than this threshold are considered outliers.

    Returns:
        numpy.ndarray: Boolean mask indicating outlier pixels (True for outliers, False for inliers).
    """
    # Define a kernel for 8 nearest neighbors

    kernel = np.ones((3, 3), dtype=bool)

    # Use a median filter to calculate median value of neighbors
    median = ndimage.median_filter(data, footprint=kernel, mode='reflect')

    # Calculate relative deviation of each pixel from its 8 neighbors' median
    # Avoid division by zero by replacing zero median with a small value
    deviation = np.abs((data - median)/(median + 1e-12))

    # Create a boolean mask for pixels deviating more than the threshold
    outlier_mask = deviation > threshold

    return outlier_mask


def detector_angle(telemetry, degrees=True):
    """
    Determine the angle between the y-axis of SunCET and solar north.

    Parameters:
        telemetry (dict): a dictionary of single point values from SunCET
        degrees (bool): set to True for degrees, false for radians

   Returns:
       float: the angle between solar north and y-axis of the SunCET
    """

    # Define the observation time
    observation_time = Time(telemetry['obs_time'], scale='utc')

    # Calculate T (Julian centuries since J2000.0)
    julian_centuries = (observation_time.jd - 2451545.0) / 36525

    # Solar north's approximate RA and Dec in ICRS
    ra_north = Angle(286.13 + 0.00694 * julian_centuries, unit=u.deg)
    dec_north = Angle(63.87 - 0.00272 * julian_centuries, unit=u.deg)

    # Create the solar north vector in ICRS coordinates
    solar_north_icrs = ICRS(ra=ra_north, dec=dec_north)

    # Convert this to the GCRS frame, which aligns closely with J2000 ECI
    solar_north_gcrs = solar_north_icrs.transform_to(GCRS(obstime=observation_time))

    solar_north_eci = np.array(solar_north_gcrs.cartesian.get_xyz())

    # Quaternion spacecraft body vector
    q_body_wrt_eci = [telemetry['adcs_att_det_q_body_wrt_eci1'], telemetry['adcs_att_det_q_body_wrt_eci2'],
                      telemetry['adcs_att_det_q_body_wrt_eci3'], telemetry['adcs_att_det_q_body_wrt_eci4']]

    # Convert the quaternion to a rotation matrix
    q_body_rotation_matrix = R.from_quat(q_body_wrt_eci).as_matrix()

    # Transform the solar north vector to the CubeSat body frame
    solar_north_body = q_body_rotation_matrix @ solar_north_eci

    # if the radiator is on -Y for cooling issues
    if telemetry['plus_y_is_up']:
        detector_north = np.array([0, 1, 0])  # y-axis in body frame
    else:
        detector_north = np.array([0, -1, 0])  # -y-axis in body frame

    # Calculate the dot product and angle
    dot_product = np.dot(solar_north_body, detector_north)
    angle_rad = np.arccos(dot_product / (np.linalg.norm(solar_north_body) * np.linalg.norm(detector_north)))

    # Convert angle from radians to degrees
    if degrees:
        suncet_angle = np.degrees(angle_rad)
    else:
        suncet_angle = angle_rad

    return suncet_angle

def CROTA_2_WCSrotation_matrix(crota2, decimals=6):
    """
    converts from the Rotation of the horizontal and vertical axes in the xy-plane to WCS coordinate rotation matrix.
    :param crota2: degree rotation of the xy coordinate plane
    :param decimals: Number of decimals to keep for finite precision
    :return: pc1_1, pc1_2, pc2_1, pc2_2
    """

    pc1_1 = np.round(np.cos(np.deg2rad(crota2)), decimals=decimals)
    pc1_2 = np.round(-1 * np.sin(np.deg2rad(crota2)), decimals=decimals)
    pc2_1 = np.round(np.sin(np.deg2rad(crota2)), decimals=decimals)
    pc2_2 = np.round(np.cos(np.deg2rad(crota2)), decimals=decimals)

    return pc1_1, pc1_2, pc2_1, pc2_2

def save_to_fits(smap, filename, metadata_dict=None):
    """
    Save a SunPy Map to a FITS file, optionally including extra metadata.

    Parameters:
    -----------
    smap : sunpy.map.Map
        The SunPy map object containing image data and metadata.
    filename : str
        Path where the FITS file will be written.
    metadata_dict : dict, optional
        Dictionary of additional FITS-compatible metadata to add to the header.
        Only scalar types (str, int, float, bool) will be written.
    """
    header = smap.meta.copy()
    data = smap.data

    if metadata_dict: #TODO verify functionality with new metadata class
        for key, value in metadata_dict.items():
            if isinstance(value, (str, int, float, bool)):
                fits_key = key.upper()[:8]
                try:
                    header[fits_key] = value
                except (ValueError, TypeError, VerifyError) as e:
                    print(f"Warning: Could not add '{fits_key}' to header: {e}")

    hdu = fits.PrimaryHDU(data=data, header=fits.Header(header))
    hdu.writeto(filename, overwrite=True)
