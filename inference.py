import cv2
import numpy as np

import torch
from numpy import random

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device
import glob
import os
import json

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
        formate_result = []
        obj_count = 0
        
        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        for i, det in enumerate(det_result):

            if len(det):

                for *xyxy, conf, cls in reversed(det):

                    label = names[int(cls)]

                    formate_result.append({'score': conf,
                                                 'pt': xyxy,
                                                 "class_id": cls,
                                                 'type': label,
                                                 'obj_count':obj_count,})

                    obj_count += 1

        return formate_result

def bubble_sort(det_result):
    for i in range(len(det_result)-1):
        for j in range(i+1,len(det_result)):
            if det_result[i]['score'] > det_result[j]['score']:
                temp = det_result[i]
                det_result[i] = det_result[j]
                det_result[j] = temp

    det_result.reverse()
    return det_result

def recovery_dataset():
    
    temp_annos_json = json.load(open("data_gc/胃部高风险病变误报图片_empty/annotations/temp_crop_instances_default.json", "r"))
    
    images =[]
    temp_img = temp_annos_json['images'][0]
    img_id = 1
    
    if 'roi' in temp_img:
        del temp_img['roi']
    
    annotations = []
    temp_ann = temp_annos_json['annotations'][0]
    ann_id = 1
    
    data_dir = "data_gc/胃部高风险病变误报图片_empty"
    
    gastroDiseaseDetector = GastroDiseaseDetect(agnostic_nms = True)
    
    gastroDiseaseDetector.ini_model(model_dir="out/WJ_V1_with_mfp7-22-2_retrain/yolov7-wj_v1_with_fp/weights/last.pt")
    
    image_dirs=[]#= glob.glob(data_dir+"/temp_recovery_images/*.jpg") #获得所有图像的路径，可以替换
    
    image_dir_json = json.load(open("data_gc/胃部高风险病变误报图片_empty/annotations/crop_instances_default_empty.json", "r"))
    
    for img in image_dir_json['images']:
        image_dirs.append(os.path.join(data_dir,'crop_images',img['file_name']))
    
    #vis_dir = data_dir+"/temp_recovery_images_vis"
    
    #os.makedirs(vis_dir,exist_ok=True)
    
    dyn_conf = 0.2
    dyn_nms_iou = 0.2
    
    for image_dir in image_dirs:
    
        image = cv2.imread(image_dir)
        
        temp_img['id'] = img_id
        temp_img['file_name'] = image_dir.replace(os.path.join(data_dir,'crop_images/'),'')#os.path.basename(image_dir)
        temp_img['height'] = image.shape[0]
        temp_img['width'] = image.shape[1]
        
        images.append(temp_img.copy())
        
        result = gastroDiseaseDetector.predict(image,formate_result=True,dyn_conf=dyn_conf,dyn_nms_iou=dyn_nms_iou)
        
        if len(result):

            result = bubble_sort(result)
            max_fp = 2 if len(result)>2 else len(result)
            
            for det_id in range(max_fp):
                temp_ann['image_id'] = img_id
                temp_ann['id'] = ann_id
                temp_ann['bbox'] = [int(result[det_id]['pt'][0]),int(result[det_id]['pt'][1]),
                                    int(result[det_id]['pt'][2]-result[det_id]['pt'][0]),int(result[det_id]['pt'][3]-result[det_id]['pt'][1])]
                
                annotations.append(temp_ann.copy())
                ann_id+=1
        else:
            temp_ann['image_id'] = img_id
            temp_ann['id'] = ann_id
            temp_ann['bbox'] = [1,1,
                                int(image.shape[1]),int(image.shape[0])]
            
            annotations.append(temp_ann.copy())
            ann_id+=1
            
        img_id += 1
        #result = gastroDiseaseDetector(image,False)
        #gastroDiseaseDetector.show_result_on_image(image,result,os.path.join(vis_dir,os.path.basename(image_dir)))    
    temp_annos_json['images'] = images
    temp_annos_json['annotations'] = annotations
    
    with open(os.path.join(data_dir,'annotations/crop_instances_default.json'), 'w') as f:
        json.dump(temp_annos_json, f,ensure_ascii=False)  
    
#use case
def use_case():
    data_dir = "data_gc/10_videos_fp"
    
    gastroDiseaseDetector = GastroDiseaseDetect(agnostic_nms = True)
    
    gastroDiseaseDetector.ini_model(model_dir="out/WJ_V1_with_mfp7-22-2_retrain/yolov7-wj_v1_with_fp/weights/last.pt")
    
    image_dirs = glob.glob(data_dir+"/temp_recovery_images/*.jpg")
    
    vis_dir = data_dir+"/temp_recovery_images_vis"
    
    os.makedirs(vis_dir,exist_ok=True)
    
    dyn_conf = 0.2
    dyn_nms_iou = 0.2
    
    for image_dir in image_dirs:
    
        image = cv2.imread(image_dir)
        
        result = gastroDiseaseDetector.predict(image,formate_result=False,dyn_conf=dyn_conf,dyn_nms_iou=dyn_nms_iou)
        #print(result)
        # or 
        #result = gastroDiseaseDetector(image,False)
        #print(result)

        gastroDiseaseDetector.show_result_on_image(image,result,os.path.join(vis_dir,os.path.basename(image_dir)))    

if __name__ == '__main__':
    recovery_dataset()


