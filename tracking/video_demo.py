import os
import sys
import argparse
import cv2

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import Tracker


def select_roi(videofile):
    """Allow user to select ROI from the first frame of the video.
    args:
        videofile: Path to the video file
    returns:
        tuple: (x, y, w, h) coordinates of selected ROI
    """
    cap = cv2.VideoCapture(videofile)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return None
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return None
    
    # Create window and allow ROI selection
    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI("Select ROI", frame, False)
    cv2.destroyAllWindows()
    cap.release()
    
    return roi

def run_video(tracker_name, tracker_param, videofile, optional_box=None, debug=None, save_results=False):
    """Run the tracker on your webcam.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        debug: Debug level.
    """
    # If no optional box is provided, allow user to select ROI
    if optional_box is None:
        print("No bounding box provided. Please select ROI in the video window...")
        optional_box = select_roi(videofile)
        if optional_box is None:
            print("No ROI selected. Exiting...")
            return
    
    tracker = Tracker(tracker_name, tracker_param, "video")
    tracker.run_video(videofilepath=videofile, optional_box=optional_box, debug=debug, save_results=save_results)


def main():
    parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
    parser.add_argument('--tracker_name', type=str, default='ostrack', help='Name of tracking method.')
    parser.add_argument('--tracker_param', type=str, default='vitb_384_mae_ce_32x4_got10k_ep100', help='Name of parameter file.')
    parser.add_argument('--videofile', type=str, default='videos/8.mp4', help='path to a video file.')
    parser.add_argument('--optional_box', type=float, default=None, nargs="+", help='optional_box with format x y w h.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--save_results', dest='save_results', action='store_true', default=True, help='Save bounding boxes')   
    parser.set_defaults(save_results=False)

    args = parser.parse_args()

    run_video(args.tracker_name, args.tracker_param, args.videofile, args.optional_box, args.debug, args.save_results)


if __name__ == '__main__':
    main()
