from yolov7gastroscopy.inference import *
import time

gastro_disease_detector = GastroDiseaseDetect(half =True)

gastro_disease_detector.ini_model(model_dir="/data2_fast/zzhang/yolov7/runs/train/yolov76/weights/best.pt")

image = cv2.imread("IMG_01.00280655.0023.13592300330.jpg")

for i in range(100):

    t1 = time.time()

    result = gastro_disease_detector.predict(image, formate_result = False)
    # or
    # result = gastroDiseaseDetector(image, formate_result = False)

    t2 = time.time()
    print(f'({(1E3 * (t2 - t1)):.1f}ms) Inference')


gastro_disease_detector.show_result_on_image(image,result,'test_result.jpg')