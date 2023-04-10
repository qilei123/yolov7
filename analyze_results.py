import json
import cv2
import os

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
    
if __name__ == "__main__":
    #visualize(visual_cats={'gt_neges':['pd_poses','pd_neges'],'pd_neges':['gt_poses','gt_neges']})
    tally_recall_precision()
    pass