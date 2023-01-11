import fiftyone as fo
import fiftyone.zoo as foz
import os

name = 'gastro_cancer_datasets'
#data_path = '/data2/qilei_chen/DATA/2021_2022gastro_cancers/2022_2/crop_images/'
#labels_path = '/data2/qilei_chen/DATA/2021_2022gastro_cancers/2022_2/annotations/crop_instances_default.json'

def get_db1(dataset_dir_index = 0):
    dataset_dirs = ["/data3/qilei_chen/DATA/gastro8-12/协和21-11月~2022-5癌变已标注/协和2021-11月_2022-5癌变_20221121", #该批数据用于测试
                    "/data3/qilei_chen/DATA/gastro8-12/2021-2022年癌变已标注/20221111/2021_2022_癌变_20221111/",
                    "/data3/qilei_chen/DATA/gastro8-12/低级别_2021_2022已标注/2021_2022_低级别_20221110/",
                    "/data3/qilei_chen/DATA/gastro8-12/协和2022_第一批胃早癌视频裁图已标注/20221115/癌变2022_20221115",
                    "/data3/qilei_chen/DATA/gastro8-12/协和2022_第二批胃早癌视频裁图已标注/协和_2022_癌变_2_20221117"]

    images_folder_name = 'crop_images/'
    annotations_dir = 'annotations/crop_instances_default.json'
    
    data_path = os.path.join(dataset_dirs[dataset_dir_index],images_folder_name)
    labels_path = os.path.join(dataset_dirs[dataset_dir_index],annotations_dir)
    return data_path,labels_path

def get_db2(db_set = 'train'):
    append_fp_data_dir = "/data2/qilei_chen/wj_fp_images1"
    
    assert db_set in ['train','test'],"db_set not exists!"
    
    fp_ann_file = 'fp_instances_default_'+db_set+'.json'
    
    return append_fp_data_dir,os.path.join(append_fp_data_dir,fp_ann_file)

def get_db3():
    append_fp_data_dir = "/home/ycao/DEVELOPMENTS/yolov7/data_gc/xiangya_far_2021/crop_images"
    
    fp_ann_file = '/home/ycao/DEVELOPMENTS/yolov7/data_gc/xiangya_far_2021/annotations/crop_instances_default.json'
    
    return append_fp_data_dir,fp_ann_file

#data_path,labels_path = get_db1(3)
#data_path,labels_path = get_db2()
data_path,labels_path = get_db3()



#dataset = fo.Dataset.from_images_dir("/data2/zinan_xiong/gastritis_0906_v1")
dataset = fo.Dataset.from_dir(
        dataset_type = fo.types.COCODetectionDataset,
        data_path = data_path,
        labels_path = labels_path,
        )
#dataset = fo.load_dataset('erosiveulcer_fine')

session = fo.launch_app(dataset, remote=True)
session.wait()
