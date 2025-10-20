import cv2
import numpy as np

def nothing(x):
    pass

def color_calibrator(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    # image = cv2.resize(image, (0, 0), fx=2, fy=2)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    # create trackbars for color change
    cv2.createTrackbar('H_lower', 'image', 0, 179, nothing)
    cv2.createTrackbar('S_lower', 'image', 0, 255, nothing)
    cv2.createTrackbar('V_lower', 'image', 0, 255, nothing)
    cv2.createTrackbar('H_upper', 'image', 179, 179, nothing)
    cv2.createTrackbar('S_upper', 'image', 255, 255, nothing)
    cv2.createTrackbar('V_upper', 'image', 255, 255, nothing)

    while True:
        # get current positions of four trackbars
        h_lower = cv2.getTrackbarPos('H_lower', 'image')
        s_lower = cv2.getTrackbarPos('S_lower', 'image')
        v_lower = cv2.getTrackbarPos('V_lower', 'image')
        h_upper = cv2.getTrackbarPos('H_upper', 'image')
        s_upper = cv2.getTrackbarPos('S_upper', 'image')
        v_upper = cv2.getTrackbarPos('V_upper', 'image')

        lower_bound = np.array([h_lower, s_lower, v_lower])
        upper_bound = np.array([h_upper, s_upper, v_upper])

        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        
        # Display the resulting frame
        cv2.imshow('image', mask)
        
        k = cv2.waitKey(1) & 0xFF
        if k == 27: # press 'esc' to exit
            break

    cv2.destroyAllWindows()
    

def detect_red_objects(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    # image = cv2.resize(image, (0, 0), fx=2, fy=2)
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for red color in HSV based on calibration
    lower_red = np.array([130, 72, 26])
    # lower_red = np.array([113, 82, 96])
    upper_red = np.array([179, 255, 255])
    
    # lower_red = np.array([101, 98, 97])
    # upper_red = np.array([179, 255, 255])

    # Create a mask for red color
    mask = cv2.inRange(hsv_image, lower_red, upper_red)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around detected objects
    for contour in contours:
        # Get the bounding box
        x, y, w, h = cv2.boundingRect(contour)
        # Draw the bounding box
        if w > 10 and h > 10:  # Filter out small contours
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Detected Red Objects', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return image

def detect_red_objects_box(image_path, display=False):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    # image = cv2.resize(image, (0, 0), fx=2, fy=2)
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for red color in HSV based on calibration
    lower_red = np.array([170, 120, 123])
    # lower_red = np.array([130, 72, 26])
    # lower_red = np.array([113, 82, 96])
    upper_red = np.array([179, 255, 255])
    
    # lower_red = np.array([101, 98, 97])
    # upper_red = np.array([179, 255, 255])

    # Create a mask for red color
    mask = cv2.inRange(hsv_image, lower_red, upper_red)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bbox_list = []
    for contour in contours:
        # Get the bounding box
        x, y, w, h = cv2.boundingRect(contour)
        # Draw the bounding box
        if w > 10 and h > 10:  # Filter out small contours
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            bbox_list.append((x, y, w, h))

    # Display the result
    if display:
        cv2.imshow('Detected Red Objects', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return bbox_list


if __name__ == '__main__':
    # color_calibrator("red_object/12.png")
    boxes = detect_red_objects_box("red_object/12.png", display=True)
    
    # print(boxes)