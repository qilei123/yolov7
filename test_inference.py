from yolov7gastroscopy.inference import *
import time

gastro_disease_detector = GastroDiseaseDetect(half =True)

gastro_disease_detector.ini_model(model_dir="out/WJ_V1_with_mfp7-22-2_retrain/yolov7-wj_v1_with_fp/weights/best.pt")

img_dir = 'data_gc/gc_df2/crop_images/190/00008_20211015_170416_425.jpg'

image = cv2.imread(img_dir)

iter = 1
for i in range(iter):

    t1 = time.time()

    result = gastro_disease_detector.predict(image, formate_result = False)
    # or
    # result = gastroDiseaseDetector(image, formate_result = False)

    t2 = time.time()
    print(f'({(1E3 * (t2 - t1)):.1f}ms) Inference')


gastro_disease_detector.show_result_on_image(image,result,'results/test_result4.jpg',[0])