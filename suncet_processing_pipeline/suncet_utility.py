"""
Helper functions for the processing pipeline
"""
import astropy.units as u
from astropy.coordinates import ICRS, GCRS, Angle
from astropy.time import Time
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation as R


def detect_outliers(data, threshold=500):
    """
    Detects pixels in the input 2D array that deviate significantly from their 8 nearest neighbors.

    Parameters:
        data (numpy.ndarray): Input 2D array (image) containing pixel values.
        threshold (float): Threshold for deviation from neighbors. Pixels deviating
                          more than this threshold are considered outliers.

    Returns:
        numpy.ndarray: Boolean mask indicating outlier pixels (True for outliers, False for inliers).
    """
    # Define a kernel for 8 nearest neighbors

    kernel = np.ones((3, 3), dtype=bool)

    # Use a median filter to calculate median value of neighbors
    median = ndimage.median_filter(data, footprint=kernel, mode='reflect')

    # Calculate absolute deviation of each pixel from its 8 neighbors' median
    deviation = np.abs(data - median)

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
