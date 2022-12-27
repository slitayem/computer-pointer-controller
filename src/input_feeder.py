"""
Class for feeding input from webcam capture, video or image to the model
"""
import os
import logging

import cv2


class InputFeeder:
    """
    Helper class for feeding input from
        webcam video capture, video or image to the model
    """
    def __init__(self, src_input=0):
        """
        The input data source depends on the src param
        src_input: 0 -> webcam capture
            string -> path to the input file (video or image)
        """
        self.image_mode = False
        if isinstance(src_input, str):
            if not os.path.exists(src_input):
                raise FileNotFoundError(f"The file {src_input} doesn't exist.")
            images_extensions = ['.jpg', '.bmp', '.png']
            self.image_mode = list(
                filter(str(src_input).endswith, images_extensions)) != []
    
        if self.image_mode:
            logging.info("Reading feed from provided image")
            self.capture = cv2.imread(src_input)
            self.info = {
                "height": self.capture.shape[0],
                "width": self.capture.shape[1],
                "channels": self.capture.shape[2]}
        else:
            self.capture = cv2.VideoCapture(src_input)
            capture = "Webcam" if src_input == 0 else src_input
            logging.info(f"Reading feed from video {capture}")
            if not self.capture.isOpened():
                raise IOError(f"Cannot open video {src_input}")
            self.info = {
                "framecount": self.capture.get(cv2.CAP_PROP_FRAME_COUNT),
                "fps": self.capture.get(cv2.CAP_PROP_FPS),
                "width": int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "codec": int(self.capture.get(cv2.CAP_PROP_FOURCC))
            }

    def next_batch(self, batch_size=10, quit_key="q"):
        """
        Returns the next image from either a video file or webcam.
        If image_mode is set to True it will return the same ima
        """
        if self.image_mode:
            if self.capture is not None:
                yield self.capture
            else:
                return
        else:
            while self.capture.isOpened():
                try:
                    for _ in range(batch_size):
                        ret, frame = self.capture.read()
                        if not ret:
                            # no frames has been grabbed
                            raise StopIteration
                        yield frame
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord(quit_key):
                            break
                except StopIteration:
                    logging.debug("End of the captured feed.")
                    return

    def close(self):
        """ Closes video file or capturing device."""
        if not self.image_mode:
            self.capture.release()
