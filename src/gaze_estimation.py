"""
Gaze Estimation Model
@author: Saloua Litayem
"""
import time
import numpy as np

from model import BaseModel


class GazeEstimator(BaseModel):
    """
    Class for theGaze Estimation Model.
    """
    model_name = "gaze-estimation-adas-0002"
    model_src = "intel"
    def __init__(self, model, device='CPU', batch_size=1):
        super().__init__(model, device, batch_size)

    def preprocess_input(self, image: np.ndarray,  input_name: str=None):
        """
        Preprocess input image and eyes images
        """
        left_eye, right_eye, head_pose_angles = image
        # shapes BxCxHxW
        # B - batch size   C - number of channels
        # H - image height W - image width
        # eyes shape: 1x3x60x60
        # head pose: 1x3
        left_eye_img = super().preprocess_input(left_eye, 'left_eye_image')
        right_eye_img = super().preprocess_input(right_eye, 'right_eye_image')
        return {
            'left_eye_image': left_eye_img,
            'right_eye_image': right_eye_img,
            'head_pose_angles': head_pose_angles
        }

    def predict(self, image: np.ndarray):
        """perform prediction"""
        net_input = self.preprocess_input(image)
        start_inference = time.time()
        output = self.exec_network.infer(net_input)
        self.inference_durations.append(int((time.time() - start_inference) * 1000))
        output = self.preprocess_output(output, net_input['head_pose_angles'])
        return output

    def preprocess_output(self, outputs, head_position):
        """
        Pre-processing output
        """
        # Cartesian coordinates of the gaze direction vector
        # gaze_vector: 1x3
        # roll = head_position[2]
        # output = outputs[self.output_name][0, :]
        # gaze_vector = output / cv2.norm(output)

        # cos_value = math.cos(roll * math.pi / 180.0)
        # sin_value = math.sin(roll * math.pi / 180.0)

        # x = gaze_vector[0] * cos_value * gaze_vector[1] * sin_value
        # y = gaze_vector[0] * sin_value * gaze_vector[1] * cos_value
        # return (x, y)
        return outputs[self.output_name][0, :2]