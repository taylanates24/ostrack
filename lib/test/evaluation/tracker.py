import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv

from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np


def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids = None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]



def select_roi(videofile, select_red=False, frame=None):
    """Allow user to select ROI from the first frame of the video.
    args:
        videofile: Path to the video file
    returns:
        tuple: (x, y, w, h) coordinates of selected ROI
    """
    a_cap = None
    if frame is None:
        a_cap = cv.VideoCapture(videofile)
        if not a_cap.isOpened():
            print("Error: Could not open video file")
            return None

        ret, frame = a_cap.read()
        if not ret:
            print("Error: Could not read first frame")
            a_cap.release()
            return None

    # Create window and allow ROI selection

    if select_red:
        roi = detect_red_objects_box(frame, display=False)
    else:
        cv.namedWindow("Select ROI", cv.WINDOW_NORMAL)
        cv.resizeWindow("Select ROI", 1152, 864)
        roi = cv.selectROI("Select ROI", frame, False)
        cv.destroyAllWindows()

    if a_cap:
        a_cap.release()

    return roi


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

    def create_tracker(self, params):
        tracker = self.tracker_class(params, self.dataset_name)
        return tracker

    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()

        tracker = self.create_tracker(params)

        output = self._track_sequence(tracker, seq, init_info)
        return output

    def _track_sequence(self, tracker, seq, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        output = {'target_bbox': [],
                  'time': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            if len(seq.ground_truth_rect) > 1:
                info['gt_bbox'] = seq.ground_truth_rect[frame_num]
            out = tracker.track(image, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def run_video(self, videofilepath, optional_box=None, debug=None, save_results=False, bounding_box_thickness=2, bounding_box_color=(0, 255, 0), output_fps=20.0, detect_red=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []
        output_video_path = 'output_video.mp4'
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        # cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        # cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        # cv.imshow(display_name, frame)
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        out_video = cv.VideoWriter(output_video_path, fourcc, output_fps, (frame_width, frame_height)) # 20.0 is the fps
        init_box = None
        init_area = None
        
        def _build_init_info(box):
            return {'init_bbox': box}

        
        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            print("No bounding box provided. Please select ROI in the video window...")
            
                
            optional_box = select_roi(videofilepath, select_red=detect_red)
            if optional_box is None:
                print("No ROI selected. Exiting...")
                return
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
            # while True:
                # # cv.waitKey()
                # frame_disp = frame.copy()

                # cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                #            1.5, (0, 0, 0), 1)

                # x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                # init_state = [x, y, w, h]
                # init_area = w * h  # Store initial bbox area
                # tracker.initialize(frame, _build_init_info(init_state))
                # output_boxes.append(init_state)
                # break


        while True:
            ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            # Draw box
            if init_box is None:
                init_box = output_boxes[0]
                init_area = init_box[2] * init_box[3]  
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)

            # Calculate current bbox area
            current_area = state[2] * state[3]
            
            # Only draw if current area is not more than 2x initial area
            if current_area <= 1.5 * init_area:
                cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                             bounding_box_color, bounding_box_thickness)
            else:
                new_box = select_roi(videofilepath, select_red=detect_red, frame=frame)
                print("new box is selected!")
                if new_box:
                    state = list(new_box)
                    init_box = state
                    init_area = state[2] * state[3]
                    tracker.initialize(frame, _build_init_info(state))
                    output_boxes.append(state)
                    cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                                 bounding_box_color, bounding_box_thickness)
                
            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            out_video.write(frame_disp)
            cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)

            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)

                cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                init_area = w * h  # Store initial bbox area
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()
        out_video.release()
        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')


    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        params = param_module.parameters(self.parameter_name)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")


import cv2
def detect_red_objects_box(image, display=False):
    # Read the image
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for red color in HSV based on calibration
    # lower_red = np.array([121, 150, 93])
    # # lower_red = np.array([170, 120, 123])
    # # lower_red = np.array([113, 82, 96])
    # upper_red = np.array([179, 255, 255])
    
    
    lower_red = np.array([118, 92, 25])
    # lower_red = np.array([130, 72, 26])
    # lower_red = np.array([113, 82, 96])
    upper_red = np.array([179, 255, 109])
    
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
            
            bbox_list.append([x, y, w, h])

    if not bbox_list:
        return None
    
    biggest_bbox = max(bbox_list, key=lambda x: x[2] * x[3])
    
    x, y, w, h = biggest_bbox
    padding_factor = 1.5
    new_w = w * padding_factor
    new_h = h * padding_factor
    new_x = x - (new_w - w) / 2
    new_y = y - (new_h - h) / 2
    
    padded_bbox = [int(new_x), int(new_y), int(new_w), int(new_h)]
    
    cv2.rectangle(image, (int(new_x), int(new_y)), (int(new_x + new_w), int(new_y + new_h)), (0, 255, 0), 2)
    if display:
        cv2.imshow('Detected Red Objects', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return padded_bbox