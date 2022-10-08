import cv2
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

# This class is to predict diseases on gastro images.

class GastroDiseaseDetect():
    def __init__(self, img_size = 640, conf = 0.3, nms_iou = 0.5, gpu_id = 0): #initialize hyperparameters
        self.img_size = img_size
        self.conf = conf
        self.nms_iou = nms_iou
        self.gpu_id = gpu_id

    def ini_model(self, model_dir:str): #load model through model dir
        self.device = select_device(self.gpu_id)
        self.model = attempt_load(model_dir, map_location=self.device)  # load FP32 model

    def predict(self, image:np.ndarray, dyn_conf = 0.3, dyn_nms_iou = 0.5): #image is a single cv2 image
        """
        图像AI分析
        Args:
            image (np.ndarray): 待分析的图像​
        Returns:
            dict: 返回分析结果数据
            示例:
            {
                0: {
                    'score': tensor(0.5932),
                    'pt': tensor([353.1492, 144.6515, 424.8212, 248.7242]),
                    'type': 'AP'
                },
                1: { ... }
            }
            0:表示编号，后面即为对应的数据；
            score:为该分数，分数越高表示可能性越大；
            pt:为坐标；
            type:0: erosive 1:ulcer 2:cancer 3: others
        """
        return None

    def formate_result(self,det_result): #transfer predict results into required formate
        pass

#use case
if __name__ == '__main__':

    gastroDiseaseDetector = GastroDiseaseDetect()
    
    gastroDiseaseDetector.ini_model(model_dir="")
    
    image = cv2.imread("test.jpg")
    
    result = gastroDiseaseDetector.predict(image)

