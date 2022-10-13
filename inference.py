import cv2
import numpy as np

import torch
from numpy import random

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device

# This class is to predict diseases on gastro images.

class GastroDiseaseDetect():
    def __init__(self, img_size = 640, conf = 0.3,  
                gpu_id = 0, nms_iou = 0.5, agnostic_nms = False): #initialize hyperparameters
        self.img_size = img_size
        self.conf = conf
        # set the gpu id
        self.gpu_id = gpu_id
        # config for nms
        self.nms_iou = nms_iou
        self.agnostic_nms = agnostic_nms

        self.device = select_device(str(self.gpu_id))

    def ini_model(self, model_dir:str): #load model through model dir

        #self.model = attempt_load(model_dir, map_location=self.device)  # load FP32 model
        
        self.model = torch.load(model_dir, map_location=self.device)  # load
        self.model = self.model['ema' if self.model.get('ema') else 'model'].float().fuse().eval()

        self.stride = int(self.model.stride.max())  # model stride
        
        self.img_size = check_img_size(self.img_size, s=self.stride)  # check img_size



    def predict(self, image:np.ndarray, dyn_conf:float = 0.3, 
                    dyn_nms_iou:float = 0.5, formate_result:bool = True): #image is a single cv2 image
        """
        图像AI分析
        Args:
            image (np.ndarray): 待分析的图像​
            dyn_conf (float,0-1): 动态置信阈值
            dyn_nms_iou (float,0-1): 动态nms_iou阈值,用于限制重叠框的接受度
        Returns:
            dict: 返回分析结果数据
            示例:
            {
                0: {
                    'score': tensor(0.5932),
                    'pt': tensor([353.1492, 144.6515, 424.8212, 248.7242]),
                    "class_id": tensor(0),
                    'type': 'AP'
                },
                1: { ... }
            }
            0:表示编号，后面即为对应的数据；
            score:为该分数，分数越高表示可能性越大；
            pt:为坐标；
            class_id: id number for each category
            type:0: erosive 1:ulcer 2:cancer 3: others
        """
        self.conf = dyn_conf
        assert self.conf>=0 and self.conf<1, "conf should be in the range [0,1)"

        self.nms_iou = dyn_nms_iou
        assert self.nms_iou>=0 and self.nms_iou<1, "nms_iou should be in the range [0,1)"

        # Padded resize
        img = letterbox(image, self.img_size, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xWxH
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)        

        pred = self.model(img)[0]
        
        pred = non_max_suppression(pred, self.conf, self.nms_iou, agnostic=self.agnostic_nms)
        
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to image size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
        
        if formate_result:
            return self.__formate_result(pred)
        
        return pred

    def __call__(self, image: np.ndarray,formate_result:bool = True):
        return self.predict(image, formate_result = formate_result)

    def show_result_on_image(self, image, pred, save_img:str = ''):
        gn = torch.tensor(image.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        if save_img:
            for i, det in enumerate(pred):
                if len(det):
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, image, label=label, color=colors[int(cls)], line_thickness=1)

            cv2.imwrite(save_img, image)

    #private for result formate
    def __formate_result(self,det_result): #transfer predict results into required formate
        formate_result = dict()
        obj_count = 0
        
        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        for i, det in enumerate(det_result):

            if len(det):

                for *xyxy, conf, cls in reversed(det):

                    label = names[int(cls)]

                    formate_result[obj_count] = {'score': conf,
                                                 'pt': xyxy,
                                                 "class_id": cls,
                                                 'type': label}

                    obj_count += 1

        return formate_result

#use case
if __name__ == '__main__':

    gastroDiseaseDetector = GastroDiseaseDetect()
    
    gastroDiseaseDetector.ini_model(model_dir="/data/qilei/DATASETS/WJ_V1/yolov7_single_cls_2/yolov7x-wj_v1/weights/best.pt")
    
    image = cv2.imread("/data/qilei/DATASETS/WJ_V1/images/3/IMG_01.00279277.0009.09195700180.jpg")
    
    result = gastroDiseaseDetector.predict(image)
    # or
    # result = gastroDiseaseDetector(image)
    
    gastroDiseaseDetector.show_result_on_image(image,result,'test.jpg')

