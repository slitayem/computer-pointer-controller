"""
Head Pose Estimation Model
@author: Saloua Litayem
"""
from model import BaseModel
import images


class HeadPoseEstimator(BaseModel):
    """
    Class for the Head Pose Estimation Model.
    """
    model_name = "head-pose-estimation-adas-0001"
    model_src = "intel"
    def __init__(self, model, device='CPU', batch_size=1):
        super().__init__(model,
            device, batch_size)

    def preprocess_output(self, outputs, image):
        """
        Process head pose estimation output
        : return array with then yaw, pitch and roll
        """
        yaw = outputs["angle_y_fc"][0, 0]
        pitch = outputs["angle_p_fc"][0, 0]
        roll = outputs["angle_r_fc"][0, 0]
        images.write_text(image,
            text= 'yaw:{:.2f} | pitch:{:.2f}'.format(yaw, pitch),
            xpos=10, ypos=20, color='red')
        images.write_text(image, text='roll:{:.2f}'.format(roll),
            xpos=10, ypos=30, color='red')
        return [yaw, pitch, roll]