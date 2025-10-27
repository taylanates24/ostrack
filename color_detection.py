import cv2
import numpy as np
import time
import paho.mqtt.client as mqtt
import sys
import json
import argparse


# --- MQTT Configuration ---
BROKER_ADDRESS = "broker.hivemq.com"
BROKER_PORT = 1883
TOPIC_PATH = "tracking/vector"  # Use a unique topic path


# --- MQTT Callback Functions ---
def on_connect(client, userdata, flags, rc):
    """
    Called upon successful connection or connection failure.
    rc (return code) 0 means success.
    """
    if rc == 0:
        print(f"Connected to MQTT Broker at {BROKER_ADDRESS}:{BROKER_PORT}!")
    else:
        print(f"Failed to connect, return code {rc}")
        sys.exit(f"Connection error: {rc}")


def on_publish(client, userdata, mid):
    """
    Called when the message is successfully published.
    """
    print(f"Message ID {mid} published successfully.")


# --- MQTT Client Setup ---
def setup_mqtt_client():
    """Sets up the client, connects, and starts the loop."""
    client = mqtt.Client(client_id="VectorPublisherClient")

    # 1. Assign callback functions
    client.on_connect = on_connect
    client.on_publish = on_publish

    # 2. Connect to the broker
    try:
        client.connect(BROKER_ADDRESS, BROKER_PORT, keepalive=60)
    except Exception as e:
        print(f"Could not connect to broker: {e}")
        return None

    # 3. Start the loop in the background.
    client.loop_start()
    return client


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
    cv2.resizeWindow('image', 640, 700)

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
    # lower_red = np.array([130, 72, 26])

    # upper_red = np.array([179, 255, 255])
    
    lower_red = np.array([167, 85, 127])
    upper_red = np.array([179, 159, 178])

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
    # cv2.imshow('Detected Red Objects', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imwrite(f"frame_drone_detected.jpg", image)
    return image

def detect_red_objects_box_from_frame(image,):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for red color in HSV based on calibration
    lower_red = np.array([161, 148, 100])
    upper_red = np.array([179, 255, 255])

    # Create a mask for red color
    mask = cv2.inRange(hsv_image, lower_red, upper_red)

    # Optional: Apply morphological operations to remove noise.
    # This can reduce the number of small, irrelevant contours found.
    # kernel = np.ones((5, 5), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


    # Find contours in the mask
    # Using cv2.RETR_EXTERNAL is faster than cv2.RETR_TREE because it only finds the parent contours
    # and doesn't build a full hierarchy, which is not used in this case.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bbox_list = []
    valid_boxes = []
    for contour in contours:
        # Get the bounding box
        x, y, w, h = cv2.boundingRect(contour)
        # Draw the bounding box
        if w > 10 and h > 10:  # Filter out small contours
            valid_boxes.append((x, y, w, h))

    if valid_boxes:
        # Find the largest bounding box by area
        largest_box = max(valid_boxes, key=lambda item: item[2] * item[3])
        x, y, w, h = largest_box
        bbox_list.append(largest_box)

    
    return bbox_list, valid_boxes

def detect_red_objects_box(image_path, display=False):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    bbox_list, all_boxes = detect_red_objects_box_from_frame(image)
    
    return bbox_list, all_boxes


def draw_arrow_from_box_to_center(frame, bbox):
    """Draws an arrow from the center of a bounding box to the center of the frame."""
    frame_height, frame_width, _ = frame.shape
    frame_center = (frame_width // 2, frame_height // 2)

    x, y, w, h = bbox
    box_center = (x + w // 2, y + h // 2)

    cv2.arrowedLine(frame, frame_center, box_center, (0, 0, 255), 4)


def get_vector_to_center(frame, bbox):
    """Calculates the vector from the frame center to the box center."""
    frame_height, frame_width, _ = frame.shape
    frame_center = (frame_width // 2, frame_height // 2)

    x, y, w, h = bbox
    box_center = (x + w // 2, y + h // 2)

    vector = (box_center[0] - frame_center[0], box_center[1] - frame_center[1])
    return vector


def process_video(video_path, output_path=None, display=False, publish_interval=50):
    
    mqtt_client = setup_mqtt_client()
    if not mqtt_client:
        print("Exiting due to MQTT connection failure.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    if output_path is not None:
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    else:
        out = None

    if display:
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video', 1280, 720)

    frame_count = 0
    while cap.isOpened():
        
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        start_time = time.time()
    
        boxes, all_boxes = detect_red_objects_box_from_frame(frame)

        end_time = time.time()
        print(f"Detection time: {(end_time - start_time) * 1000} ms")
        if boxes:
            x, y, w, h = boxes[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)
            draw_arrow_from_box_to_center(frame, boxes[0])

            if frame_count % publish_interval == 0:
                vector = get_vector_to_center(frame, boxes[0])
                payload = json.dumps({"vector_x": vector[0], "vector_y": vector[1]})
                print(f"Publishing to MQTT: {payload}")
                mqtt_client.publish(TOPIC_PATH, payload, qos=1)

        if output_path is not None:
            # Write the frame to the output video
            out.write(frame)

        # Display the result
        if display:
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        print("MQTT client disconnected.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a video to detect red objects and publish vector data.")
    parser.add_argument('--interval', type=int, default=50, help='The interval of frames to publish the vector data.')
    args = parser.parse_args()

    # color_calibrator("red_object/12.png")
    # boxes = detect_red_objects_box("red_object/12.png", display=False)
    process_video("drone/drone_red_object_1.mp4", "output_video_with_vector.mp4", display=True, publish_interval=args.interval)
    
    # print(boxes)