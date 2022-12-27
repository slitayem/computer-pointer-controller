"""
Facial Landmarks Model
@author: Saloua Litayem
"""
import cv2

import numpy as np
from model import BaseModel


class FacialLandmarksDetector(BaseModel):
    """
    Class for the Landmarks regressor model
    """
    model_name = "landmarks-regression-retail-0009"
    model_src = "intel"

    def __init__(self, model, device='CPU', batch_size=1):
        super().__init__(model, device, batch_size)

    def preprocess_output(self, outputs: np.ndarray,
        image: np.ndarray, boxsize: int =12):
        """
        Process Facial Landmark estimation output
        : return array with then yaw, pitch and roll
        """
        # flatten the array without making a copy of it
        output_array = np.stack(outputs.values()).ravel()
        assert output_array.shape == (10,)

        height = image.shape[0]
        width = image.shape[1]

        center_left = (int(output_array[0] * width),
            int(output_array[1] * height))
        center_right = (int(output_array[2] * width),
            int(output_array[3] * height))
        
        l_x0, l_y0 = tuple(map(lambda x: x - boxsize, center_left))
        l_x1, l_y1 = tuple(map(lambda x: x + boxsize, center_left))

        r_x0, r_y0 = tuple(map(lambda x: x - boxsize, center_right))
        r_x1, r_y1 = tuple(map(lambda x: x + boxsize, center_right))

        left_eye = image[l_y0:l_y1, l_x0:l_x1]
        right_eye = image[r_y0:r_y1, r_x0:r_x1]
        cv2.rectangle(image, (l_x0, l_y0), (l_x1, l_y1), (0, 255, 0), 1)
        cv2.rectangle(image, (r_x0, r_y0), (r_x1, r_y1), (0, 255, 0), 1)

        return center_left, center_right, left_eye, right_eye
