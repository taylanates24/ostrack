import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import Tracker


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


def main():
    parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
    parser.add_argument('--tracker_name', type=str, default='ostrack', help='Name of tracking method.')
    parser.add_argument('--tracker_param', type=str, default='vitb_384_mae_ce_32x4_got10k_ep100', help='Name of parameter file.')
    # parser.add_argument('--videofile', type=str, default='videos/1_ir.mp4', help='path to a video file.')
    parser.add_argument('--videofile', type=str, default='videos/red_car_2.mp4', help='path to a video file.')
    parser.add_argument('--optional_box', type=float, default=None, nargs="+", help='optional_box with format x y w h.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--save_results', dest='save_results', action='store_true', default=True, help='Save bounding boxes')
    parser.add_argument('--bounding_box_thickness', type=int, default=2, help='Thickness of bounding box')
    parser.add_argument('--bounding_box_color', type=int, default=(0, 0, 255), nargs=3, help='Color of bounding box (B G R). Default is green.')
    parser.add_argument('--output_fps', type=float, default=30, help='FPS of the output video')
    parser.set_defaults(save_results=False)

    args = parser.parse_args()

    color = tuple(args.bounding_box_color) if args.bounding_box_color else (0, 255, 0)

    run_video(args.tracker_name, args.tracker_param, args.videofile, args.optional_box, args.debug, args.save_results, args.bounding_box_thickness, color, args.output_fps)


if __name__ == '__main__':
    main()
