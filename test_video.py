from yolov7gastroscopy.inference import *
import time
import glob
import os

def CropImg(image,roi=None):
    if roi is None:
        height, width, d = image.shape

        pixel_thr = 10
        
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

    gastro_disease_detector = GastroDiseaseDetect(half =True,gpu_id=1)

    gastro_disease_detector.ini_model(model_dir="single_category.pt")

    videos_dir = '/data3/xiaolong_liang/data/videos_2022/202201_r06/gastroscopy/'

    report_images_dir = '/data2/qilei_chen/wj_fp_images1'

    video_list = glob.glob(os.path.join(videos_dir,"*.mp4"))

    

    for video_dir in video_list:
        print(videos_dir)
        video = cv2.VideoCapture(video_dir)

        video_name = os.path.basename(video_dir)

        images_folder = os.path.join(report_images_dir,video_name.replace('.mp4',''))

        if not os.path.exists(images_folder):
            os.makedirs(images_folder)

        fps = video.get(cv2.CAP_PROP_FPS)

        ret, frame = video.read()
        
        roi = None

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
            frame = gastro_disease_detector.show_result_on_image(frame,result,None)            

            video_writer.write(frame)

            ret, frame = video.read()
            frame_id+=1

def extract_frames():
    org_videos_dir = '/data3/xiaolong_liang/data/videos_2022/202201_r06/gastroscopy/'
    result_videos_dir = '/data2/qilei_chen/wj_fp_images1/'

if __name__ == '__main__':
    #process_videos()
    extract_frames()