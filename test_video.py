from yolov7gastroscopy.inference import *
import time
import glob
import os
import json

import pycocotools.coco as COCO

def CropImg(image,roi=None):
    if roi is None:
        height, width, d = image.shape

        pixel_thr = 30 #30和10针对不同的视频，10针对xl65
        
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

def process_videos():

    gastro_disease_detector = GastroDiseaseDetect(half =True,gpu_id=2)

    #gastro_disease_detector.ini_model(model_dir="single_category.pt")
    
    model_name ='WJ_V1_with_mfp4-3'
    
    gastro_disease_detector.ini_model(model_dir='out/'+model_name+'/yolov7-wj_v1_with_fp/weights/best_f2.pt')

    #videos_dir = '/data3/xiaolong_liang/data/videos_2022/202201_r06/gastroscopy/'
    videos_dir = '/data3/qilei_chen/DATA/gastro_cancer_tests/xiehe2111_2205/'

    #report_images_dir = '/data2/qilei_chen/wj_fp_images1'
    report_images_dir = '/data3/qilei_chen/DATA/gastro_cancer_tests/xiehe2111_2205_'+model_name+'/'
    
    os.makedirs(report_images_dir,exist_ok=True)

    video_list = glob.glob(os.path.join(videos_dir,"*.mp4"))
    
    roi = None

    for video_dir in sorted(video_list):
        print(video_dir)
        video = cv2.VideoCapture(video_dir)

        video_name = os.path.basename(video_dir)

        images_folder = os.path.join(report_images_dir,video_name.replace('.mp4',''))

        os.makedirs(images_folder,exist_ok=True)

        fps = video.get(cv2.CAP_PROP_FPS)
        
        video.set(cv2.CAP_PROP_POS_FRAMES,1000)

        ret, frame = video.read()
        
        video.set(cv2.CAP_PROP_POS_FRAMES,0)

        if roi==None:
            _, roi = CropImg(frame)

        size = (int(roi[2]-roi[0]),int(roi[3]-roi[1]))

        video_writer = cv2.VideoWriter(os.path.join(report_images_dir,os.path.basename(video_dir)+'.avi'), 
                                        cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size)

        frame_id_report_log = open(os.path.join(report_images_dir,os.path.basename(video_dir)+'.txt'),'w')

        frame_id = 0
        while ret:

            frame = CropImg(frame,roi)

            result = gastro_disease_detector.predict(frame, formate_result = False)
            
            report = False
            for i, det in enumerate(result):
                if len(det):
                    report = True
            if report:
                frame_id_report_log.write(str(frame_id)+'\n')
            
            cv2.putText(frame, str(frame_id), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            frame = gastro_disease_detector.show_result_on_image(frame,result,visible_ids=[0])            

            video_writer.write(frame)

            ret, frame = video.read()
            frame_id+=1

def extract_frames():
    org_videos_dir = '/data3/xiaolong_liang/data/videos_2022/202201_r06/gastroscopy/'
    result_videos_dir = '/data2/qilei_chen/wj_fp_images1/'

    record_list = glob.glob(os.path.join(result_videos_dir,"*.txt"))

    for record_file in record_list:

        print(record_file)

        record = open(record_file)

        org_video = cv2.VideoCapture(os.path.join(org_videos_dir,os.path.basename(record_file.replace(".txt",""))))

        result_video = cv2.VideoCapture(record_file.replace("txt","avi"))

        frame_id = record.readline()

        while frame_id:
            frame_id = int(frame_id)
            
            def read_and_save_frame(cap,save_dir,folder,frame_id):
                cap.set(cv2.CAP_PROP_POS_FRAMES,frame_id)
                suc, org_frame = cap.read()
                os.makedirs(os.path.join(save_dir.replace(".mp4.txt",""),folder),exist_ok=True)
                if suc:
                    cv2.imwrite(os.path.join(save_dir.replace(".mp4.txt",""),folder,str(frame_id).zfill(10)+'.jpg'),org_frame)
            
            #save for the original image
            read_and_save_frame(org_video,record_file,'org',frame_id)

            #save for the result image
            read_and_save_frame(result_video,record_file,'result',frame_id)

            frame_id = record.readline()



def reprocess_images():
    

    gastro_disease_detector = GastroDiseaseDetect(half =False,gpu_id=1)

    gastro_disease_detector.ini_model(model_dir="/data1/qilei_chen/DEVELOPMENTS/yolov7/out/WJ_V1_with_mfp1/yolov7-wj_v1_with_fp/weights/best.pt")

    folder_list = glob.glob('/data2/qilei_chen/wj_fp_images1/images/*')

    for folder_dir in folder_list:
        print(folder_dir)
        images_folder = os.path.join(folder_dir,'org')
        reprocess_folder = os.path.join(folder_dir,'WJ_V1_with_mfp1')
        os.makedirs(reprocess_folder,exist_ok=True)

        images_list = sorted(glob.glob(os.path.join(images_folder,'*.jpg')))
        
        roi = None
        
        for image_dir in images_list:
            image = cv2.imread(image_dir)

            if roi==None:
                _, roi = CropImg(image)
            else:
                image = CropImg(image,roi)

            result = gastro_disease_detector.predict(image, formate_result = False)
            
            report = False  

            for i, det in enumerate(result):
                if len(det):
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if int(cls)==0:
                            report = True

            if report:
                image = gastro_disease_detector.show_result_on_image(image,result,None) 
                cv2.imwrite(os.path.join(reprocess_folder,os.path.basename(image_dir)),image)

def t2s(t): #t format is hh:mm:ss
    if t != '0':
        h,m,s = t.strip().split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
    else:
         return 0

def parse_periods(periods_file = None):
    #periods_file = open('/data2/qilei_chen/DATA/2021_2022gastro_cancers/2021_videos/periods.txt')
    line = periods_file.readline()
    periods = {}
    while line:
        elements = line.split('	')
        periods[elements[0]]=[]
        for e in elements[1::2]:
            if e=='':
                break
            ts = e.split('-')
            periods[elements[0]].append([t2s(ts[0]),t2s(ts[1])])
        
        line = periods_file.readline()

    return periods

def checkPeriods(periods,fps,frame_id):
    
    for period in periods:
        if frame_id>period[0]*fps and frame_id<period[1]*fps:
            return False

    return True

def process_videos_xiangya():

    gastro_disease_detector = GastroDiseaseDetect(half =False,gpu_id=2)

    train_dataset_name = 'WJ_V1_with_mfp3-1'

    gastro_disease_detector.ini_model(model_dir="out/"+train_dataset_name+"/yolov7-wj_v1_with_fp/weights/best.pt")

    videos_dir = '/data2/qilei_chen/DATA/2021_2022gastro_cancers/2022_videos/'

    report_images_dir = os.path.join(videos_dir,train_dataset_name)

    report_videos_dir = os.path.join(report_images_dir,'result_videos')

    os.makedirs(report_videos_dir,exist_ok=True)

    video_list = sorted(glob.glob(os.path.join(videos_dir,"*.mp4")))

    #video_list = parse_periods(open(os.path.join(videos_dir,'periods.txt')))

    for video_dir in video_list[13:]:
        video_dir = os.path.join(videos_dir,video_dir)

        print(video_dir)
        
        video = cv2.VideoCapture(video_dir)

        video_name = os.path.basename(video_dir)

        images_folder = os.path.join(report_images_dir,video_name.replace('.mp4',''))
        org_images_folder = os.path.join(images_folder,'org')
        #os.makedirs(org_images_folder,exist_ok=True)
        process_images_folder = os.path.join(images_folder,'process')
        #os.makedirs(process_images_folder,exist_ok=True)

        fps = video.get(cv2.CAP_PROP_FPS)

        ret, frame = video.read()
        
        roi = None

        if roi==None:
            _, roi = CropImg(frame)

        size = (int(roi[2]-roi[0]),int(roi[3]-roi[1]))

        video_writer = cv2.VideoWriter(os.path.join(report_videos_dir,os.path.basename(video_dir)+'.avi'), 
                                        cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size)

        #frame_id_report_log = open(os.path.join(report_images_dir,os.path.basename(video_dir)+'.txt'),'w')

        frame_id = 0
        while ret:

            frame = CropImg(frame,roi)

            result = gastro_disease_detector.predict(frame, formate_result = False)
            
            report = False  

            for i, det in enumerate(result):
                if len(det):
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if int(cls)==0:
                            report = True
            #report  = report and checkPeriods(video_list[video_name],fps,int(frame_id))
            if report:
                #frame_id_report_log.write(str(frame_id)+'\n')
                #cv2.imwrite(os.path.join(org_images_folder,str(frame_id).zfill(10)+'.jpg'),frame)
                pass
            #cv2.putText(frame, str(frame_id), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            frame = gastro_disease_detector.show_result_on_image(frame,result,None,[0]) 

            if report:
                #cv2.imwrite(os.path.join(process_images_folder,str(frame_id).zfill(10)+'.jpg'),frame)
                pass           

            video_writer.write(frame)

            ret, frame = video.read()
            frame_id+=1

def generate_fp_coco():
    gastro_disease_detector = GastroDiseaseDetect(half =False,gpu_id=1)
    gastro_disease_detector.ini_model(model_dir="/data1/qilei_chen/DEVELOPMENTS/yolov7/out/WJ_V1_with_mfp1/yolov7-wj_v1_with_fp/weights/best.pt")

    data_dir = '/data2/qilei_chen/DATA/2021_2022gastro_cancers/2021_videos'

    folder_list = sorted(glob.glob(os.path.join(data_dir,'WJ_V1_with_mfp1_filted/*')))

    #temp_coco = COCO('/data2/qilei_chen/DATA/2021_2022gastro_cancers/2022_1/annotations/crop_instances_default.json')
    temp_coco = json.load(open('/data2/qilei_chen/DATA/2021_2022gastro_cancers/2022_1/annotations/crop_instances_default.json'))

    temp_coco['categories'] = [{'id':1,'name':'others','supercategory':''}]

    images = []

    annotations =[]

    temp_image = temp_coco['images'][0].copy()

    temp_annotation = temp_coco['annotations'][0].copy()

    image_id = 1

    annotation_id = 1

    for folder_dir in folder_list:
        print(folder_dir)

        video = cv2.VideoCapture(os.path.join(data_dir,os.path.basename(folder_dir)+".mp4"))

        ret, frame = video.read()
        
        roi = None

        if roi==None:
            _, roi = CropImg(frame)

        images_folder = os.path.join(folder_dir,'process')

        images_list = sorted(glob.glob(os.path.join(images_folder,'*.jpg')))

        for image_dir in images_list:
            
            image_dir = image_dir.replace('WJ_V1_with_mfp1_filted','WJ_V1_with_mfp1')
            image_dir = image_dir.replace('process','org')

            frame_id = int(os.path.basename(image_dir).replace('.jpg',''))

            video.set(cv2.CAP_PROP_POS_FRAMES,frame_id)

            ret, image = video.read()

            image = CropImg(image,roi)
            
            #image = cv2.imread(image_dir)

            temp_image['file_name'] = image_dir.replace(data_dir,'')

            temp_image['id'] = image_id

            temp_image['height'],temp_image['width'],_ = image.shape

            temp_image['roi'] = roi

            result = gastro_disease_detector.predict(image, formate_result = False)

            temp_annotation_id = annotation_id

            report = True

            for i, det in enumerate(result):
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    temp_annotation['id'] = annotation_id
                    temp_annotation['bbox'] = [int(xyxy[0]),int(xyxy[1]),int(xyxy[2]-xyxy[0]),int(xyxy[3]-xyxy[1])]
                    temp_annotation['category_id'] = 1
                    temp_annotation['image_id'] = image_id
                    temp_annotation['segmentation'] = [[]]
                    temp_annotation['area'] = temp_annotation['bbox'][2]* temp_annotation['bbox'][3]
                    annotations.append(temp_annotation.copy())
                    if int(cls)==0:
                        report = False
                    annotation_id+=1
                    
            if temp_annotation_id == annotation_id or report:
                print(image_dir)
            images.append(temp_image.copy())
            image_id+=1

    temp_coco['images'] = images
    temp_coco['annotations'] = annotations

    with open(os.path.join(data_dir,"fp_instances_default_train.json"), "w") as outfile:
        json.dump(temp_coco,outfile)

if __name__ == '__main__':
    process_videos()
    #extract_frames()
    #reprocess_images()
    #print(parse_periods())
    #process_videos_xiangya()
    #generate_fp_coco()
    pass