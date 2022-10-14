from yolov7gastroscopy.inference import *
import time

gastro_disease_detector = GastroDiseaseDetect(half =True)

gastro_disease_detector.ini_model(model_dir="/data/qilei/DATASETS/WJ_V1/yolov7_single_cls_2/yolov7x-wj_v1/weights/best.pt")

image = cv2.imread("/data/qilei/DATASETS/WJ_V1/images/3/IMG_01.00279277.0009.09195700180.jpg")

for i in range(100):

    t1 = time.time()

    result = gastro_disease_detector.predict(image, formate_result = False)
    # or
    # result = gastroDiseaseDetector(image, formate_result = False)

    t2 = time.time()
    print(f'({(1E3 * (t2 - t1)):.1f}ms) Inference')


gastro_disease_detector.show_result_on_image(image,result,'test.jpg')