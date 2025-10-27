import numpy as np

def calculate_pixel_to_meter_ratio(drone_altitude_meters, width_pixels=3840, height_pixels=2160, horizontal_fov_degrees=81.0):
    """
    Calculates how many meters on the ground plane correspond to 1 pixel (Ground Sample Distance - GSD).

    This function is based on the assumption that the camera is completely parallel 
    to the ground (Pitch=0, Roll=0) and the ground is flat.

    Parameters:
    - drone_altitude_meters (float): The drone's altitude from the ground (e.g., 50.0).
    - width_pixels (int): Horizontal resolution (e.g., 3840).
    - height_pixels (int): Vertical resolution (e.g., 2160).
    - horizontal_fov_degrees (float): The camera's horizontal field of view (e.g., 81.0).

    Returns:
    - tuple: (meters_per_pixel_horizontal, meters_per_pixel_vertical)
    """

    # 1. Calculate Focal Length in Pixels (f_pixel)
    horizontal_fov_radians = np.deg2rad(horizontal_fov_degrees)
    
    # Horizontal Focal Length: tan(FOV/2) = (W/2) / f_pixel_h
    focal_length_pixels_horizontal = (width_pixels / 2) / np.tan(horizontal_fov_radians / 2)

    # Vertical Focal Length: Assuming aspect ratio is preserved (f_v = f_h * H/W)
    focal_length_pixels_vertical = focal_length_pixels_horizontal * (height_pixels / width_pixels)

    # 2. Calculate the Scaling Factor (GSD)
    # GSD = drone_altitude_meters / f_pixel
    
    meters_per_pixel_horizontal = drone_altitude_meters / focal_length_pixels_horizontal
    meters_per_pixel_vertical = drone_altitude_meters / focal_length_pixels_vertical

    return (meters_per_pixel_horizontal, meters_per_pixel_vertical)

def convert_to_geographical(vector, angle_degrees, width_pixels, height_pixels, drone_altitude_meters, horizontal_fov_degrees):
    """
    Converts a Cartesian vector from a drone camera frame to geographical distance
    components (North-South, East-West) in meters.

    Args:
        vector (tuple): (x, y) coordinates from the drone's frame.
                        (+X = Right/East, +Y = Up/North).
        angle_degrees (float): The counter-clockwise rotation angle required to
                               align the input frame's North direction (which is +Y)
                               with True North.
        width_pixels (int): The width of the video frame in pixels.
        height_pixels (int): The height of the video frame in pixels.
        drone_altitude_meters (float): The altitude of the drone in meters.
        horizontal_fov_degrees (float): The horizontal field of view of the camera in degrees.

    Returns:
        tuple: A tuple containing:
            ns_meters (float): The magnitude of the North-South component in meters.
            ns_direction (str): "North" or "South", or "" if magnitude is near zero.
            ew_meters (float): The magnitude of the East-West component in meters.
            ew_direction (str): "East" or "West", or "" if magnitude is near zero.
    """
    angle_radians = np.deg2rad(angle_degrees)
    x, y = vector
    
    # Define an explicit tolerance for highly accurate comparison against zero.
    TOLERANCE = 1e-9

    # --- Step 1 & 2: Apply Standard Rotation (Counter-Clockwise) ---
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    
    # Standard 2D rotation matrix for a Y-Up frame:
    x_rot = x * cos_angle - y * sin_angle
    y_rot = x * sin_angle + y * cos_angle

    # --- Step 3: Convert pixel vector to meters ---
    # The GSD calculation must use the parameters provided to the function.
    meters_per_pixel_horizontal, meters_per_pixel_vertical = calculate_pixel_to_meter_ratio(
        drone_altitude_meters, width_pixels, height_pixels, horizontal_fov_degrees
    )
    
    # Convert rotated pixel distance to meters
    # x_rot (East-West) uses horizontal GSD
    # y_rot (North-South) uses vertical GSD
    ns_meters_signed = y_rot * meters_per_pixel_vertical
    ew_meters_signed = x_rot * meters_per_pixel_horizontal

    # --- Step 4: Decompose into Geographical Components with Direction ---
    
    # North-South Component (+y_rot is North)
    if abs(ns_meters_signed) < TOLERANCE:
        ns_meters = 0.0
        ns_direction = ""
    elif ns_meters_signed > 0:
        ns_meters = ns_meters_signed
        ns_direction = "North"
    else: # ns_meters_signed < 0
        ns_meters = -ns_meters_signed
        ns_direction = "South"
        
    # East-West Component (+x_rot is East)
    if abs(ew_meters_signed) < TOLERANCE:
        ew_meters = 0.0
        ew_direction = ""
    elif ew_meters_signed > 0:
        ew_meters = ew_meters_signed
        ew_direction = "East"
    else: # ew_meters_signed < 0
        ew_meters = -ew_meters_signed
        ew_direction = "West"

    return ns_meters, ns_direction, ew_meters, ew_direction

def run_tests():
    print("--- Running Geographical Conversion Tests (Pixel-to-Meter) ---")
    
    # Define common drone/camera parameters
    W = 3840
    H = 2160
    FOV = 81.0
    ALT = 50.0
    
    # Pre-calculate GSDs for expected values
    GSD_H, GSD_V = calculate_pixel_to_meter_ratio(ALT, W, H, FOV)

    # Helper function to check the mixed-type result (4-tuple)
    def check_result(result, expected, test_name, vector):
        # Expected/Result tuple is: (NS Meters, NS Dir, EW Meters, EW Dir)
        
        # Check numerical components (indices 0 and 2) using np.isclose
        ns_mag_pass = np.isclose(result[0], expected[0])
        ew_mag_pass = np.isclose(result[2], expected[2])
        num_pass = ns_mag_pass and ew_mag_pass
        
        # Check string components (indices 1 and 3)
        ns_dir_pass = result[1] == expected[1]
        ew_dir_pass = result[3] == expected[3]
        str_pass = ns_dir_pass and ew_dir_pass
        
        assert num_pass and str_pass, f"{test_name} failed:\n  Input: {vector}\n  Expected: {expected}\n  Got: {result}"
        print(f"{test_name} (Input: {vector}) -> Result {result} (Pass)")


    # Test 1: Pure East Vector, No Rotation (Using user's requested vector: 500, 0)
    # Input: (500 East, 0 North). Angle: 0 degrees.
    # Expected: 500 * GSD_H East, 0 North/South
    vector = (500, 0)
    angle = 0
    expected_ns_meters = 0.0
    expected_ew_meters = 500 * GSD_H
    expected = (expected_ns_meters, "", expected_ew_meters, "East")
    result = convert_to_geographical(vector, angle, W, H, ALT, FOV)
    check_result(result, expected, "Test 1 (500, 0, 0°)", vector)

    # Test 2: Pure North Vector, No Rotation
    # Input: (0 East, 1000 North). Angle: 0 degrees.
    # Expected: 1000 * GSD_V North, 0 East/West
    vector = (0, 1000)
    angle = 0
    expected_ns_meters = 1000 * GSD_V
    expected_ew_meters = 0.0
    expected = (expected_ns_meters, "North", expected_ew_meters, "")
    result = convert_to_geographical(vector, angle, W, H, ALT, FOV)
    check_result(result, expected, "Test 2 (0, 1000, 0°)", vector)

    # Test 3: Pure North Vector, 90 degrees CCW rotation
    # Input: (0 East, 1000 North). Angle: 90 degrees.
    # Rotated Vector: (-1000 East, 0 North) -> 1000 West
    # Expected: 0 North/South, 1000 * GSD_H West
    vector = (0, 1000)
    angle = 90
    expected_ns_meters = 0.0
    expected_ew_meters = 1000 * GSD_H
    expected = (expected_ns_meters, "", expected_ew_meters, "West")
    result = convert_to_geographical(vector, angle, W, H, ALT, FOV)
    check_result(result, expected, "Test 3 (0, 1000, 90°)", vector)

    # Test 4: Equal components, 180 degrees CCW rotation
    # Input: (500 East, 500 North). Angle: 180 degrees.
    # Rotated Vector: (-500 East, -500 North) -> 500 South, 500 West
    # Expected: 500 * GSD_V South, 500 * GSD_H West
    vector = (500, 500)
    angle = 180
    expected_ns_meters = 500 * GSD_V
    expected_ew_meters = 500 * GSD_H
    expected = (expected_ns_meters, "South", expected_ew_meters, "West")
    result = convert_to_geographical(vector, angle, W, H, ALT, FOV)
    check_result(result, expected, "Test 4 (500, 500, 180°)", vector)
    
    # Test 5: Complex vector, 45 degrees rotation
    # Input: (100 East, -100 South). Angle: 45 degrees.
    # Rotated Vector: (141.42 East, 0 North) -> 141.42 East
    vector = (100, -100)
    angle = 45
    
    # Calculate expected rotation components manually:
    # x_rot = 100 * cos(45) - (-100) * sin(45) = 100*sqrt(2)/2 + 100*sqrt(2)/2 = 100 * sqrt(2)
    # y_rot = 100 * sin(45) + (-100) * cos(45) = 100*sqrt(2)/2 - 100*sqrt(2)/2 = 0
    
    EW_PIXELS = 100 * np.sqrt(2)
    
    expected_ns_meters = 0.0
    expected_ew_meters = EW_PIXELS * GSD_H
    expected = (expected_ns_meters, "", expected_ew_meters, "East")
    result = convert_to_geographical(vector, angle, W, H, ALT, FOV)
    check_result(result, expected, "Test 5 (100, -100, 45°)", vector)


    print("--- All tests passed! ---")

if __name__ == '__main__':
    run_tests()