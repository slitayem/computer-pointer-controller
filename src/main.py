"""
@author: Saloua Litayem
Mouse pointer controller
"""
import os
from argparse import ArgumentParser
from pprint import pprint
import time
import json

import cv2

from input_feeder import InputFeeder
from mouse_controller import MouseController
import logger
from face_detection import FaceDetector
from head_pose_estimation import HeadPoseEstimator
from gaze_estimation import GazeEstimator
from facial_landmarks_detection import FacialLandmarksDetector
import images

APP_DIR = os.path.expandvars("$HOME/app-artifacts")
LOGS_DIR = os.path.join(APP_DIR, "logs")
IMAGES_EXTENSIONS = ['.jpg', '.bmp']
MODEL_NAME = os.getenv('MODEL_NAME')
PERF_FOLDER = os.path.join(APP_DIR, "perf")

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--models_dir", required=True, type=str,
                        help="Models directory")
    parser.add_argument("-i", "--input", type=str, default=0,
                        help="Path to image or video file. Using the webcam capture if this argument is not provided")
    parser.add_argument("-p", "--precision", type=str, default="FP32",
                    help="Floating-point precision e.g. FP32")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-db", "--debug", action='store_true', default=False,
                    help="Set to use the app in debug mode."
                    "(False by default)")
    return parser


def init_app():
    """ Initialze the app """
    global MOUSE_CONTROLLER
    args = build_argparser().parse_args()

    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR, exist_ok=True)
    if not os.path.exists(PERF_FOLDER):
        os.makedirs(PERF_FOLDER, exist_ok=True)

    _logger = logger.Logger()
    os.environ['LOGGING_LEVEL'] = "DEBUG" if args.debug else "INFO"
    _logger.setup_logging()
    # init the mouse controller
    MOUSE_CONTROLLER = MouseController(mouse_precision=12, mouse_speed=0.05)
    MOUSE_CONTROLLER.move_to(100, 300)
    return args


def get_model_path(
    models_dir: str, model_name: str,
    precision: float, model_src: str='intel'):
    """construct the model path given its name"""
    return os.path.join(models_dir, model_src,
        model_name, precision, f"{model_name}.xml")


def init_model_from_name(models_dir,
    device, precision,
    model_classname, arc_view=False):
    """
    Initialize model object from provided name
    """
    class_ = globals()[model_classname]
    path = get_model_path(models_dir,
        getattr(class_, 'model_name'),
        precision, getattr(class_, 'model_src'))
    model_obj = class_(
        path, device)
    model_arc_view = model_obj.load_model()
    if arc_view:
        logger.info("---- {}: Architecture:".format(model_arc_view))
    return model_obj


def init_models(args):
    """
    Initialize all the models objects
    """
    global face_detector, head_pose_estimator, \
        facial_landmarks_detector, gaze_estimator

    face_detector = init_model_from_name(args.models_dir,
        args.device, args.precision, 'FaceDetector')
    head_pose_estimator = init_model_from_name(args.models_dir,
        args.device, args.precision, 'HeadPoseEstimator')
    gaze_estimator = init_model_from_name(args.models_dir,
        args.device, args.precision, 'GazeEstimator')
    facial_landmarks_detector = init_model_from_name(args.models_dir,
        args.device, args.precision, 'FacialLandmarksDetector')


def main():
    """
    Load the network and parse the output.
    """
    args = init_app()
    init_models(args)

    window_name: str = "Video capture"
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 300, 300)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    feed = InputFeeder(args.input)

    out = cv2.VideoWriter("output.avi",
        cv2.VideoWriter_fourcc(*'avc1'),
        feed.info.get('fps', 15),
        (feed.info["width"], feed.info["height"]), True)
    models_stats = []
    start_time = time.time()
    frames_count = 0
    for _, frame in enumerate(feed.next_batch(1)):
        # 1. Face detection
        point1, point2, confidence = face_detector.predict(frame)
        cropped_face = frame[point1[1]:point2[1], point1[0]:point2[0]]
        drawed_image = images.draw_bbox(frame,
            point1, point2,
            f"Face: {confidence * 100 :.2f}%",
            color="green", thickness=2)

        # 2. Facial Landmarks detection
        center_left_eye, center_right_eye, left_eye, right_eye = \
            facial_landmarks_detector.predict(cropped_face)

        # 3. Head pose estimation
        head_pose_angles = head_pose_estimator.predict(cropped_face)

        # 4. Gaze estimation
        x_pos, y_pos = gaze_estimator.predict((left_eye, right_eye, head_pose_angles))
        x_left, y_left = center_left_eye
        x_right, y_right = center_right_eye

        cv2.arrowedLine(cropped_face, (x_left, y_left),
            (int(x_left + x_pos * 90), int(y_left + y_pos * -90)), (0, 0, 255), 2)
        cv2.arrowedLine(cropped_face, (x_right, y_right),
            (int(x_right + x_pos * 90), int(y_right + y_pos * -90)), (0, 0, 255), 2)
        cv2.imshow(
            window_name,
            drawed_image)
        out.write(frame)
        frames_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        MOUSE_CONTROLLER.move(x_pos, y_pos)
    inference_time = round(time.time() - start_time, 1)
    fps = int(frames_count / inference_time)
    print("Total inference time:", inference_time)
    print("FPS Frames/Seconds", fps)
    feed.close()
    out.release()
    with open(f'media/benchmark_{args.device}_{args.precision}.json',
        'w', encoding='utf-8') as file_:
        json.dump(facial_landmarks_detector.model_stats, file_, indent=4)
        json.dump(face_detector.model_stats, file_, indent=4)
        json.dump(head_pose_estimator.model_stats, file_, indent=4)
        json.dump(gaze_estimator.model_stats, file_, indent=4)

if __name__ == '__main__':
    main()
