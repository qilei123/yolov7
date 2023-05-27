from yolov7gastroscopy.inference import *
from y7models.yolo import Model
from y7utils.torch_utils import intersect_dicts
import time
import pickle
import glob,os

def test_case():
    gastro_disease_detector = GastroDiseaseDetect(conf = 0.25,half =True,gpu_id=3)

    #model_dir = 'single_category_y7.pt'
    #model_dir = 'multi_categories_y7.pt'
    #model_dir = 'binary_categories_y7-1-3.pt'
    model_dir = 'binary_categories_y7-22-2.pt'
    #model_dir = '3categories_y7-22-2-31-v-150.pt'
    gastro_disease_detector.ini_model(model_dir)

    img_rel_dir = 'data_gc/AI-TEST-fujian'
    
    img_dir_list = glob.glob(os.path.join(img_rel_dir,'images','*.jpg'))
    
    test_save_dir = os.path.join(img_rel_dir,'test_save_3.2')
    os.makedirs(test_save_dir,exist_ok=True)
    
    for img_dir in img_dir_list:
        image = cv2.imread(img_dir)
        
        t1 = time.time()

        result = gastro_disease_detector.predict(image, formate_result = False)
        
        # or
        # result = gastroDiseaseDetector(image, formate_result = False)

        t2 = time.time()
        print(f'({(1E3 * (t2 - t1)):.1f}ms) Inference')


        gastro_disease_detector.show_result_on_image(image,result,os.path.join(test_save_dir,os.path.basename(img_dir)))

def transfer_model(src,dst):
    '''
    before run this transfer function, should add the module mapping in serialization.py 
    load_module_mapping: Dict[str, str] = {
        # See https://github.com/pytorch/pytorch/pull/51633
        'torch.tensor': 'torch._tensor',
        "models.yolo": "y7models.yolo",
        "models.common": "y7models.common"
    }   
    ''' 
    model1 = torch.load(src)
    torch.save(model1,dst) 

if __name__ == "__main__":
    #transfer_model('single_category.pt','single_category_y7.pt')
    #transfer_model('multi_categories.pt','multi_categories_y7.pt')
    #transfer_model('out/WJ_V1_with_mfp1-3/yolov7-wj_v1_with_fp/weights/best.pt','binary_categories_y7-1-3.pt')
    #transfer_model('out/WJ_V1_with_mfp7-12/yolov7-wj_v1_with_fp/weights/best.pt','binary_categories_y7-12-0.pt')
    #transfer_model('out/WJ_V1_with_mfp7-22-2_retrain/yolov7-wj_v1_with_fp/weights/best.pt','binary_categories_y7-22-2.pt')
    #transfer_model('27_yolov7_output/WJ_V1_with_mfp7-22-2-31-v/yolov7-wj_v1_with_fp/weights/epoch_150.pt','3categories_y7-22-2-31-v-150.pt')
    test_case()
    pass