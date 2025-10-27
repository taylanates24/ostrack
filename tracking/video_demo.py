import os
import sys
import argparse
import cv2
import json
import paho.mqtt.client as mqtt
import numpy as np

prj_path = os.path.join(os.path.dirname(__file__), '..')

if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation.tracker import detect_red_objects_box

from lib.test.evaluation import Tracker

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
    magnitude = np.sqrt(vector[0]**2 + vector[1]**2)
    vector = vector / magnitude
    vector [0] = round(vector[0], 5)
    vector [1] = round(vector[1], 5)
    magnitude = round(magnitude, 1)
    return vector, magnitude

def build_init_info(box):
    return {'init_bbox': box}


def detect_and_get_box(frame, display=False):
    """Detects red objects and returns the bounding box and a detection flag."""
    box = detect_red_objects_box(frame, display=display)
    box_detected = box is not None
    return box, box_detected


def reinitialize_tracker(tracker, frame, box):
    """Re-initializes the tracker with a new bounding box."""
    state = list(box)
    tracker.init_box = state
    tracker.init_area = state[2] * state[3]
    tracker.tracker.initialize(frame, build_init_info(state))
    tracker.output_boxes.append(state)



def run_video(tracker_name, tracker_param, videofile, optional_box=None, debug=None, save_results=False, bounding_box_thickness=2, bounding_box_color=(0, 255, 0), output_fps=20.0):
    """Run the tracker on your webcam.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        debug: Debug level.
    """
    
    tracker = Tracker(tracker_name, tracker_param, "video")
    
    tracker.run_video(videofilepath=videofile, 
                      debug=debug, 
                      optional_box=optional_box,
                      save_results=save_results, 
                      bounding_box_thickness=bounding_box_thickness, 
                      bounding_box_color=bounding_box_color, 
                      output_fps=output_fps,
                      detect_red=True)



def run(tracker_name, tracker_param, videofile, display=False, out_file=None, publish_interval=50, publish_mqtt=False):
    """Run the tracker on your webcam.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        debug: Debug level.
    """
    
    tracker = Tracker(tracker_name, tracker_param, "video")
    
    mqtt_client = None
    if publish_mqtt:
        mqtt_client = setup_mqtt_client()
        if not mqtt_client:
            print("Exiting due to MQTT connection failure.")
            return


    cap = cv2.VideoCapture(videofile)
    box_detected = False
    i = 1
    frame_count = 0
    if display:
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Frame', 1280, 720)
    
    if out_file is not None:
        ret, frame = cap.read()
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
    
    while True:
        
        if i % 100 == 0:
            i += 1
            
            box, box_detected = detect_and_get_box(frame, display=False)
            if not box_detected:
                continue
            
            tracker.init_box = None
            tracker.init_area = None
            tracker.output_boxes = []
            
            if box is not None:
                reinitialize_tracker(tracker, frame, box)
                i = 1
                continue
            
        ret, frame = cap.read()
        
        if not ret:
            break
        
        
        if not box_detected:
            
            box, box_detected = detect_and_get_box(frame, display=False)
            if not box_detected:
                print("No box detected! Reading next frame...")
                continue
            
            print("new box is selected!")
            
            if tracker.init_box is None and tracker.tracker is not None:
                reinitialize_tracker(tracker, frame, box)
                i = 1
        box = tracker.run(frame, box)
        
        if box is None:
            tracker.init_box = None
            tracker.init_area = None
            tracker.output_boxes = []
            box, box_detected = detect_and_get_box(frame, display=False)
            if not box_detected:
                print("No box detected! Reading next frame...")

            else:
                reinitialize_tracker(tracker, frame, box)
                i = 1
            continue
        
        if box is not None:
            cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
            cv2.putText(frame, 'Tracking!', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
            draw_arrow_from_box_to_center(frame, box)
            if publish_mqtt and frame_count % publish_interval == 0:
                vector, magnitude = get_vector_to_center(frame, box)
                payload = json.dumps({"vector_x": vector[0], "vector_y": vector[1], "magnitude": magnitude})
                print(f"Publishing to MQTT: {payload}")
                mqtt_client.publish(TOPIC_PATH, payload, qos=1)
            
        else:
            cv2.putText(frame, 'No box detected!', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)


        if display:
            cv2.imshow('Frame', frame)
        if out_file is not None:
            out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        i += 1
        frame_count += 1
        
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        print("MQTT client disconnected.")
        
    cap.release()
    if display:
        cv2.destroyAllWindows()
    if out_file is not None:
        out.release()

def main():
    parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
    parser.add_argument('--tracker_name', type=str, default='ostrack', help='Name of tracking method.')
    parser.add_argument('--tracker_param', type=str, default='vitb_256_mae_ce_32x4_ep300', help='Name of parameter file.')
    parser.add_argument('--videofile', type=str, default='drone/drone_red_object_1.mp4', help='path to a video file.')
    parser.add_argument('--out_file', type=str, default='output_video_deneme.mp4', help='path to the output video file.')
    # parser.add_argument('--videofile', type=str, default='drone/drone_red_object_1.mp4', help='path to a video file.')
    parser.add_argument('--optional_box', type=float, default=None, nargs="+", help='optional_box with format x y w h.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--save_results', dest='save_results', action='store_true', default=True, help='Save bounding boxes')
    parser.add_argument('--bounding_box_thickness', type=int, default=2, help='Thickness of bounding box')
    parser.add_argument('--bounding_box_color', type=int, default=(0, 0, 255), nargs=3, help='Color of bounding box (B G R). Default is green.')
    parser.add_argument('--output_fps', type=float, default=30, help='FPS of the output video')
    parser.add_argument('--publish_interval', type=int, default=50, help='The interval of frames to publish the vector data.')
    parser.add_argument('--publish_mqtt', action='store_true', default=True, help='Publish vector data to MQTT.')
    parser.set_defaults(save_results=False)

    args = parser.parse_args()

    color = tuple(args.bounding_box_color) if args.bounding_box_color else (0, 255, 0)

    # run_video(args.tracker_name, args.tracker_param, args.videofile, args.optional_box, args.debug, args.save_results, args.bounding_box_thickness, color, args.output_fps)
    run(args.tracker_name, args.tracker_param, args.videofile, display=True, out_file=args.out_file, publish_interval=args.publish_interval, publish_mqtt=args.publish_mqtt)

if __name__ == '__main__':
    main()
