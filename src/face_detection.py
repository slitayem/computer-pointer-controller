"""
Facial Detection Model
@author: Saloua Litayem
"""
import logging
import numpy as np

from model import BaseModel
import images

class FaceDetector(BaseModel):
    """
    Class for the Face Detection Model.
    """
    model_name = "face-detection-adas-0001"
    model_src = "intel"
    def __init__(self, model, device='CPU', batch_size=1, min_threshold=0.6):
        """
        """
        super().__init__(model, device, batch_size)

    def preprocess_output(self, outputs: np.ndarray, image: np.ndarray,
        min_threshold: float=0.6):
        """
        Process face detection output
        : return (tuple): image with detection boxes
                (point1, point2, confidence)

        """
        height, width, _ = image.shape
        # TODO return detected objects boxes sorted by (threshold and size)
        face_label_id = 1
        # Keeping only the object with the highest confidence
        output = outputs[self.output_name]
        conf_scores = output[0, 0, :, 2]
        index = np.argmax(conf_scores)
        confidence = conf_scores[index]
        label = output[0, 0, :, 1][index]
        if confidence > min_threshold and label == face_label_id:
            xmin, ymin, xmax, ymax = output[0, 0, index, 3:]
            xmin *= width
            ymin *= height
            xmax *= width
            ymax *= height
            point1 = (int(xmin), int(ymin))
            point2 = (int(xmax), int(ymax))
            return (point1, point2, confidence)
        return (None, None, None)
