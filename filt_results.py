import os
import glob



def file_results(gts_dir,res_dir):
    gt_list = glob.glob(os.path.join(gts_dir,"*"))
    re_list = glob.glob(os.path.join(res_dir,"*"))

    gt_name_list = []

    for gt_dir in gt_list:
        gt_name_list.append(os.path.basename(gt_dir))

    for re_dir in re_list:
        if not os.path.basename(re_dir) in gt_name_list:
            os.remove(re_dir)

if __name__ == "__main__":

    file_results("/data/qilei/DATASETS/WJ_V1/show_gts/test/1/", "/data/qilei/DATASETS/WJ_V1/yolov7_single_cls_2/results/1/exp/")