import os
import glob
import sys
def change_video_names():
    records = open('temp_datas/changweijing_issues1.txt',encoding="utf8")
    #map_records = open('temp_datas/changweijing_issues_filenames_map.txt','w',encoding="utf8")
    line = records.readline()
    count=0
    new_base_name = '20220301_1024_010'

    video_dir = '/data3/qilei_chen/DATA/changjing_issues/'

    while line:
        
        line = line.replace('\n','')
        line_eles = line.split('	')

        if len(line_eles[0])>0:
            new_line = line_eles[0]+" "+new_base_name+str(count)+"_"+line_eles[1]
            new_line_eles = new_line.split(' ')
            
            if os.path.exists(os.path.join(video_dir,new_line_eles[0]+".avi")):
                org_video_name = os.path.join(video_dir,new_line_eles[0]+".avi")
            else:
                org_video_name = os.path.join(video_dir,new_line_eles[0]+".mp3")
  
            os.rename(org_video_name,org_video_name.replace(new_line_eles[0],new_line_eles[1]))
            print(org_video_name)
            print(org_video_name.replace(new_line_eles[0],new_line_eles[1]))

            #map_records.write(new_line)
            count+=1

        line = records.readline()
    

def change_video_names1():
    root_dir = '/data3/qilei_chen/DATA/changjing_issues/'
    video_dirs = glob.glob(os.path.join(root_dir,'*mp4'))+glob.glob(os.path.join(root_dir,'*avi'))

    map_name_records = open(os.path.join(root_dir,'video_names_map_bk.txt'))

    #new_base_name = '20220301_1024_010'

    for count,video_dir in enumerate(video_dirs):
        video_name = os.path.basename(video_dir)
        #new_video_dir = os.path.join(root_dir,new_base_name+str(count)+'.'+video_dir[-3:])

        new_video_dir = video_dir.replace(video_dir[-4:],"."+video_dir[-3:])
        #map_name_records.write(video_name+' '+new_video_name+'\n')
        os.rename(video_dir,new_video_dir)
        print(video_dir)
        print(new_video_dir)

def change_video_names2():
    root_dir = '/data3/qilei_chen/DATA/changjing_issues/'
    video_dirs = glob.glob(os.path.join(root_dir,'*mp4'))+glob.glob(os.path.join(root_dir,'*avi'))

    map_name_records = open(os.path.join(root_dir,'video_names_map_bk.txt'))

    #new_base_name = '20220301_1024_010'
    line = map_name_records.readline()

    while line:
        eles = line[:-1].split(' ')
        eles[1] = eles[1].replace('_mp4','.mp4')
        eles[1] = eles[1].replace('_avi','.avi')
        file_dir = os.path.join(root_dir,eles[1])
        o_file_dir = os.path.join(root_dir,eles[0])
        if os.path.exists(file_dir):
            os.rename(file_dir,o_file_dir)
        line = map_name_records.readline()
import json
from pycocotools.coco import COCO
import cv2
import numpy as np

def CropImg(image,roi=None):
    if roi is None:
        height, width, d = image.shape

        pixel_thr = 30#若针对xl65数据集需要改成10
        
        w_start=0
        while True:
            if np.sum(image[int(height/2),int(w_start),:])/d>pixel_thr:
                break
            w_start+=1

        w_end=int(width-1)
        while True:
            if np.sum(image[int(height/2),int(w_end),:])/d>pixel_thr:
                break
            w_end-=1

        h_start=0
        while True:
            if np.sum(image[int(h_start),int(width/2),:])/d>pixel_thr:
                break
            h_start+=1

        h_end=int(height-1)
        while True:
            if np.sum(image[int(h_end),int(width/2),:])/d>pixel_thr:
                break
            h_end-=1

        roi = [w_start,h_start,w_end,h_end]

        #print(image[int(height-1),int(width-1),:])

        return image[roi[1]:roi[3],roi[0]:roi[2],:],roi
    else:
        return image[roi[1]:roi[3],roi[0]:roi[2],:]

def crop_wg(anno_dir = "/data2/qilei_chen/DATA/2021_2022gastro_cancers/2022_1/",_items_map = None):
    
    temp_annotation = json.load(open(os.path.join(anno_dir,'annotations/instances_default.json'),errors='ignore'))

    coco = COCO(os.path.join(anno_dir,'annotations/instances_default.json'))
    
    items_map = {"癌变":"cancer","糜烂":"erosive","溃疡":"ulcer","低级别":"low","高级别":"high"}
    if _items_map is not None:
        items_map = _items_map
    for i in range(len(temp_annotation['categories'])):
        temp_annotation['categories'][i]['name'] = items_map[temp_annotation['categories'][i]['name']]
    
    temp_annotation["images"] = []
    temp_annotation["annotations"] = []

    #fix_roi1 = [663,33,1890,1042]
    #fix_roi2 = [570,13,1600,956]
    #fix_roi3 = [173,45,688,530]
    fix_roi1 = None
    ab_count=0
    for ImgId in coco.getImgIds():    
        img = coco.loadImgs([ImgId])[0]
        
        file_dir = os.path.join(anno_dir,'images',img['file_name'])
        image = cv2.imdecode(np.fromfile(file_dir, dtype=np.uint8), -1)
        fix_roi = fix_roi1
        if fix_roi1 == None:
            crop_img,fix_roi = CropImg(image)
        else:
            crop_img = CropImg(image,fix_roi1)  

        os.makedirs(os.path.dirname(os.path.join(anno_dir,'crop_images', img['file_name'])),exist_ok=True)

        if (fix_roi[2]-fix_roi[0])/(fix_roi[3]-fix_roi[1])>1.2 or (fix_roi[2]-fix_roi[0])/(fix_roi[3]-fix_roi[1])<0.8:
            #print(fix_roi)
            #print(file_dir)
            cv2.imencode('.jpg', crop_img)[1].tofile(os.path.join(anno_dir,'crop_images', img['file_name']))
            pass
        else:
            cv2.imencode('.jpg', crop_img)[1].tofile(os.path.join(anno_dir,'crop_images', img['file_name']))
            pass
        
        #img['roi'] = roi
        '''
        if img['width']==1920:
            fix_roi = fix_roi1
            if os.path.exists(os.path.join('E:/DATASET/放大胃镜/放大胃镜图片筛选/v3_白光/crop_abnormal1/ab1',os.path.basename(img['file_name']))):
                fix_roi = fix_roi2
                print(img['file_name'])
                ab_count += 1
            crop_img = CropImg(image, fix_roi)
            cv2.imencode('.jpg', crop_img)[1].tofile(os.path.join(anno_dir,'crop', img['file_name']))
        else:
            if os.path.exists(os.path.join('E:/DATASET/放大胃镜/放大胃镜图片筛选/v3_白光/crop_abnormal1/ab2',os.path.basename(img['file_name']))):
                fix_roi = fix_roi3
                crop_img = CropImg(image, fix_roi)
                print(img['file_name'])
                ab_count += 1
                cv2.imencode('.jpg', crop_img)[1].tofile(os.path.join(anno_dir,'crop', img['file_name']))
            else:
                fix_roi = [0,0,img['width'],img['height']]
                cv2.imencode('.jpg', image)[1].tofile(os.path.join(anno_dir,'crop', img['file_name']))
        '''


        img['roi'] = fix_roi
        img['width'] = fix_roi[2]-fix_roi[0]
        img["height"] = fix_roi[3]-fix_roi[1]
        temp_annotation["images"].append(img)
        
        annIds =  coco.getAnnIds(ImgId)
        anns = coco.loadAnns(annIds)

        for ann in anns:
            ann['bbox'][0] = 0 if round(ann['bbox'][0] - fix_roi[0],2)<0 else round(ann['bbox'][0] - fix_roi[0],2)
            ann['bbox'][1] = 0 if round(ann['bbox'][1] - fix_roi[1],2)<0 else round(ann['bbox'][1] - fix_roi[1],2)
            if len(ann["segmentation"]):
                for i in range(int(len(ann["segmentation"][0])/2)):
                    ann["segmentation"][0][2*i] = 0 if round(ann["segmentation"][0][2*i]-fix_roi[0],2)<0 else round(ann["segmentation"][0][2*i]-fix_roi[0],2)
                    ann["segmentation"][0][2*i+1] = 0 if round(ann["segmentation"][0][2*i+1]-fix_roi[1],2)<0 else round(ann["segmentation"][0][2*i+1]-fix_roi[1],2)
                    
                temp_annotation["annotations"].append(ann)
            else:
                temp_annotation["annotations"].append(ann)

    print(ab_count)
    with open(os.path.join(anno_dir,'annotations/crop_instances_default.json'), 'w',errors='ignore') as outfile:
        json.dump(temp_annotation, outfile,ensure_ascii=False)
        
    return len(coco.getImgIds()),len(coco.getAnnIds())

def mv_folder():
    src_dir = '/data2/qilei_chen/wj_fp_images1/images'
    dst_dir = '/data2/qilei_chen/wj_fp_images1/images1'
    folder_list = glob.glob(os.path.join(src_dir,"2*"))

    for folder in folder_list:
        os.makedirs(os.path.join(dst_dir,os.path.basename(folder)),exist_ok=True)
        command_line = 'mv '+os.path.join(folder,'WJ_V1_with_mfp1')+' '+os.path.join(dst_dir,os.path.basename(folder))
        print(command_line)
        os.system(command_line)

def crop_ecs():
    folders = glob.glob('data_ec/user*')
    total_img_count,total_ann_count = 0,0
    for folder in folders:
        sub_folders = glob.glob(os.path.join(folder,"*"))
        sub_folders = [sub_folder for sub_folder in sub_folders if not sub_folder.endswith('.zip')]
        for sub_folder in sub_folders:
            print(sub_folder)
            img_count,ann_count = crop_wg(sub_folder,{'食管癌':'EsophagealCancer'})
            total_img_count += img_count
            total_ann_count += ann_count
            
    print(total_img_count)
    print(total_ann_count)

def procee_data_gl():
    #merge train and test sets
    merge = False
    cat_ids = range(13)
    dst_dir = "data_gl/gpcb"
    if merge:
        
        source_dir = 'data_gl/cx_data_gl/gastro_position_clasification_best'
        
        os.makedirs(dst_dir,exist_ok=True)
        for cat_id in cat_ids:
            os.makedirs(os.path.join(dst_dir,'all',str(cat_id)),exist_ok=True)
            command1 = 'cp '+os.path.join(source_dir,'train',str(cat_id),'*')+' '+os.path.join(dst_dir,'all',str(cat_id))
            os.system(command1)
            command2 = 'cp '+os.path.join(source_dir,'test',str(cat_id),'*')+' '+os.path.join(dst_dir,'all',str(cat_id))
            os.system(command2)
    sample_folder = 'sample0'
    sample_cat_ids = range(3,12)
    count_sample = 0
    count_all = 0
    fold_size = 7
    for cat_id in sample_cat_ids:
        sample_dir = os.path.join(dst_dir,sample_folder,str(cat_id))
        os.makedirs(sample_dir,exist_ok=True)
        cat_sample_list = glob.glob(os.path.join(dst_dir,'all',str(cat_id),"*"))
        for index , cat_sample in enumerate(cat_sample_list):
            if index%fold_size==0:
                command = command2 = 'cp '+cat_sample+' '+sample_dir
                count_sample+=1
                #os.system(command)
            count_all+=1
    print(count_all)
    print(count_sample)


min_value = 0.0000001

def fx(r,p,x):
    return (1+x**2)*r*p/(r+(x**2)*p+min_value)


def get_best():
    record_file_dir = '27_yolov7_output/WJ_V1_with_mfp7-22-2-28-2-v/yolov7-wj_v1_with_fp/results.txt'

    record_file = open(record_file_dir)
    
    line = record_file.readline()
    
    bf05,bf1,bf2 = 0,0,0
    bf05_e,bf1_e,bf2_e = 0,0,0
    
    br = 0
    br_e = 0
    
    br_p = 0
    br_p_e = 0
    
    e_count = 0
    while line:
        
        if line.startswith("videos"):
            
            eles = line.strip().split("\t")
            
            recall = float(eles[0].split(':')[1])
            precision = float(eles[1].split(':')[1])
            
            bf05 = bf05 if bf05>fx(recall,precision,0.5) else fx(recall,precision,0.5)
            if bf05==fx(recall,precision,0.5):
                bf05_e = e_count
            
            bf1 = bf1 if bf1>fx(recall,precision,1) else fx(recall,precision,1)
            if bf1==fx(recall,precision,1):
                bf1_e = e_count
            
            bf2 = bf2 if bf2>fx(recall,precision,2) else fx(recall,precision,2)
            if bf2==fx(recall,precision,2):
                bf2_e = e_count
                
                
            if br<recall:
                br = recall
                br_e = e_count
            
            a = 0.99    
            if br_p<a*recall+(1-a)*precision:
                br_p = a*recall+(1-a)*precision
                br_p_e = e_count
            
            e_count += 1
        
            if recall>0.25 and precision>0.81:
                print(e_count)
    
            #print('best_score:{0};{1};{2};{3},{4}'.format(bf05,bf1,bf2,br,br_p))
            #print('best:{0};{1};{2};{3};{4}'.format(bf05_e,bf1_e,bf2_e,br_e,br_p_e))
        
        line = record_file.readline()

def crop_images(imgs_dir='/data3/qilei_chen/DATA/301_1600_ade/images_data/回顾性息肉性质判断图片/白光',
                save_dir='/data3/qilei_chen/DATA/301_1600_ade/images_data/回顾性息肉性质判断图片/白光_cropped'):

    img_dir_list = glob.glob(os.path.join(imgs_dir, "*.jpg"))
    
    roi = None
    
    os.makedirs(save_dir, exist_ok=True)
    
    for img_dir in img_dir_list:
        image = cv2.imdecode(np.fromfile(img_dir, dtype=np.uint8), -1)
        if roi == None:
            image , roi = CropImg(image)
        else:
            image = CropImg(image,roi)
            
        cv2.imencode('.jpg', image)[1].tofile(os.path.join(save_dir, os.path.basename(img_dir)))

if __name__=="__main__":
    #change_video_names2()
    #crop_wg(anno_dir = "/data2/qilei_chen/DATA/2021_2022gastro_cancers/2021_1/")
    #crop_wg(anno_dir = "/data2/qilei_chen/DATA/2021_2022gastro_cancers/2021_2/")
    #crop_wg(anno_dir = "/data2/qilei_chen/DATA/2021_2022gastro_cancers/2022_1/")
    #crop_wg(anno_dir = "/data2/qilei_chen/DATA/2021_2022gastro_cancers/2022_2/")
    #mv_folder()
    #crop_wg(anno_dir= "/data3/qilei_chen/DATA/gastro8-12/2021-2022年癌变已标注/20221111/2021_2022_癌变_20221111/")
    #crop_wg(anno_dir="/data3/qilei_chen/DATA/gastro8-12/低级别_2021_2022已标注/2021_2022_低级别_20221110/")
    #crop_wg(anno_dir="/data3/qilei_chen/DATA/gastro8-12/协和21-11月~2022-5癌变已标注/协和2021-11月_2022-5癌变_20221121")
    #crop_wg(anno_dir="/data3/qilei_chen/DATA/gastro8-12/协和2022_第一批胃早癌视频裁图已标注/20221115/癌变2022_20221115")
    #crop_wg(anno_dir="/data3/qilei_chen/DATA/gastro8-12/协和2022_第二批胃早癌视频裁图已标注/协和_2022_癌变_2_20221117")
    #crop_wg(anno_dir='data_gc/xiehe_far_1')
    #crop_wg(anno_dir='data_gc/xiehe_far_2')
    #crop_wg(anno_dir='data_gc/xiangya_far_2021')
    #crop_wg(anno_dir='data_gc/xiangya_far_2022')
    #crop_wg(anno_dir='data_gc/xiangya_202209_202211')
    #crop_wg(anno_dir="data_gc/湘雅_远景_2021_2022_低级别_20230110")
    #crop_ecs()
    #crop_wg(anno_dir="data_ec/user11/201412_201912-148", _items_map={'食管癌':'EsophagealCancer',})
    #crop_wg(anno_dir="data_gc/gas12nips",_items_map={'gas12nip':'gas12nip',})
    #procee_data_gl()
    #get_best()
    #crop_images()
    pass
    