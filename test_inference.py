from yolov7gastroscopy.inference import *
import time
import glob
import os

def inference_1():
    gastro_disease_detector = GastroDiseaseDetect(half =True,gpu_id=3)

    #gastro_disease_detector.ini_model(model_dir="out/WJ_V1_with_mfp7-22-2_retrain/yolov7-wj_v1_with_fp/weights/best.pt")
    gastro_disease_detector.ini_model(model_dir="27_yolov7_output/WJ_V1_with_mfp7-22-2-22/yolov7-wj_v1_with_fp/weights/best.pt")

    loc_ids = [2,3,4,5,6,7,8,9,10]
    loc_ids = [3,4,5,6,7,8,9,10,11]
    for loc_id in loc_ids:
        img_folder = 'data_gl/cx_data_gl/gastro_position_clasification_11/train/'+str(loc_id)
        result_save_dir = "data_gl/gpc11_22/train/"+str(loc_id)
        os.makedirs(result_save_dir,exist_ok=True)

        img_dir_list = glob.glob(os.path.join(img_folder, "*.jpg"))

        #img_dir = 'data_gc/gc_df2/crop_images/190/00008_20211015_170416_425.jpg'

        for img_dir in img_dir_list:
            image = cv2.imread(img_dir)

            iter = 1
            for i in range(iter):

                t1 = time.time()

                result = gastro_disease_detector.predict(image, formate_result = False)
                # or
                # result = gastroDiseaseDetector(image, formate_result = False)

                t2 = time.time()
                #print(f'({(1E3 * (t2 - t1)):.1f}ms) Inference')


            image, positive = gastro_disease_detector.show_result_on_image_positive(image,result,'',[0])
            
            if positive:
                cv2.imwrite(os.path.join(result_save_dir,os.path.basename(img_dir)),image)

def inference_2():
    gastro_disease_detector = GastroDiseaseDetect(half =True,gpu_id=3)

    #gastro_disease_detector.ini_model(model_dir="out/WJ_V1_with_mfp7-22-2_retrain/yolov7-wj_v1_with_fp/weights/best.pt")
    
    
    #model_dir="27_yolov7_output/WJ_V1_with_mfp7-22-2-31-v/yolov7-wj_v1_with_fp/weights/epoch_150.pt"
    #model_dir='27_yolov7_output/WJ_V1_with_mfp7-22-2-33-v/yolov7-wj_v1_with_fp/weights/epoch_145.pt'
    #model_dir='27_yolov7_output/bkup/WJ_V1_with_mfp7-22-2_retrain/yolov7-wj_v1_with_fp/weights/best.pt'
    model_dir = '27_yolov7_output/WJ_V1_with_mfp7-22-2-34-v/yolov7-wj_v1_with_fp/weights/epoch_146.pt'
    
    
    model_folder = model_dir.split("/")[1]
    
    gastro_disease_detector.ini_model(model_dir=model_dir)
    
    src_image_folder = 'data_gc/AI-TEST-fujian/images'
    img_dir_list = glob.glob(os.path.join(src_image_folder,"*.jpg"))
    
    save_image_folder = 'data_gc/AI-TEST-fujian/'+model_folder
    os.makedirs(save_image_folder,exist_ok=True)

    pos_counter = 0

    for img_dir in img_dir_list:
        image = cv2.imread(img_dir)

        iter = 1
        for i in range(iter):

            t1 = time.time()

            result = gastro_disease_detector.predict(image, formate_result = False)
            # or
            # result = gastroDiseaseDetector(image, formate_result = False)

            t2 = time.time()
            #print(f'({(1E3 * (t2 - t1)):.1f}ms) Inference')


        image, positive = gastro_disease_detector.show_result_on_image_positive(image,result,'',[0])
        
        
        if positive:
            pos_counter += 1
        
            cv2.imwrite(os.path.join(save_image_folder,os.path.basename(img_dir)),image)
        
    print(pos_counter)
    
if __name__ == "__main__":
    #inference_1()
    inference_2()