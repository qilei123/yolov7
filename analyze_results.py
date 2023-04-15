import json
import cv2
import os
from PIL import Image

def visualize(result_file='runs/WJ_V1_with_mfp7-22-2_retrain/exp/best_c_criteria.json',
              visual_dir='runs/WJ_V1_with_mfp7-22-2_retrain/exp/visualizations',
              visual_cats={'gt_neges':[],'pd_neges':[]}):
    results_json = json.load(open(result_file))
    
    for result in results_json:
        
        for visual_cat in visual_cats:
            image = cv2.imread(result['file_dir'])
            if visual_cat in result:
                os.makedirs(os.path.join(visual_dir, visual_cat), exist_ok=True)
                for box in result[visual_cat]:
                    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255,0,0), 2)
                
                for combined_cat in visual_cats[visual_cat]:
                    for box in result[combined_cat]:
                        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,255), 2)                    
                    
                if len(result[visual_cat]):
                    cv2.imwrite(os.path.join(visual_dir,visual_cat,result['image_id']+".jpg"), image)
                    
def tally_recall_precision(result_file='runs/WJ_V1_with_mfp7-22-2_retrain/exp/best_c_criteria.json'):
    results_json = json.load(open(result_file))
    
    gts_count = 0
    pos_gts_count = 0
    
    pds_count = 0
    pos_pds_count = 0
    
    for result in results_json:
        gts_count = gts_count + len(result["gt_poses"]) + len(result["gt_neges"])
        pos_gts_count += len(result["gt_poses"])
        
        pds_count = pds_count + len(result["pd_poses"]) + len(result["pd_neges"])
        pos_pds_count += len(result["pd_poses"])
        
    print(gts_count)
    print(pos_gts_count)
    print(pos_gts_count/gts_count)
    print(pds_count)
    print(pos_pds_count)
    print(pos_pds_count/pds_count)


def generate_empty_fps():
    data_dir = 'data_gc/胃部高风险病变误报图片'
    all_images_dir = [os.path.join(fpathe,f).replace(os.path.join(data_dir,'crop_images/'),'') 
                      for fpathe,dirs,fs in os.walk(os.path.join(data_dir,'crop_images')) 
                      for f in fs 
                      if os.path.join(fpathe,f).endswith('.jpg')]
    
    annos = json.load(open(os.path.join(data_dir,'annotations/crop_instances_default.json')))
    imgs_with_box = [img_with_box['file_name'] for img_with_box in annos['images']]
    
    new_fp_images = []
    img_id = 1
    
    img_dict_template = annos["images"][0]
    
    if 'roi' in img_dict_template:
        del img_dict_template['roi']
    
    for img_dir in all_images_dir:
        if img_dir in imgs_with_box:
            pass
        else:
            image = Image.open(os.path.join(data_dir,'crop_images',img_dir))
            img_dict_template['file_name']=img_dir
            img_dict_template['width'] = image.width
            img_dict_template['height'] = image.height
            img_dict_template['id'] = img_id
            new_fp_images.append(img_dict_template.copy())
            img_id += 1
    annos['images'] = new_fp_images
    annos['annotations'] = []
    
    with open(os.path.join(data_dir,'annotations/crop_instances_default_empty.json'), 'w') as f:
        json.dump(annos, f,ensure_ascii=False)     
    
        
if __name__ == "__main__":
    #visualize(visual_cats={'gt_neges':['pd_poses','pd_neges'],'pd_neges':['gt_poses','gt_neges']})
    #tally_recall_precision()
    generate_empty_fps()
    pass