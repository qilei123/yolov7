import os

import glob

def periods_filter(periods,min_length = 2):
    
    for period in periods:
        if period[1]-period[0] <= min_length:
            periods.remove(period)
            
    return periods
    
def get_positive_periods(filename = 'data_gc/videos_test/xiehe2111_2205_WJ_V1_with_mfp3-0-6_best/20211123_104815_03_r02_olbs290.mp4.txt'):
    
    positive_peroids = []
    
    file_record = open(filename, "r")
    
    line = file_record.readline()
    
    period_on = False
    period_start = 0
    period_end = 0
    
    while line:
        
        line = line.split(' ')
        if period_on==False and line[1][:-1] == "#1":
            period_start = int(line[0])
            period_on=True
        elif period_on==True and line[1][:-1] == "#0":
            period_end = int(line[0])
            positive_peroids.append([period_start,period_end])
            period_on = False
        line = file_record.readline()
        
    return positive_peroids

def periods_IOU(period1,period2):
    
    p1 = period1[1]-period1[0]
    p2 = period2[1]-period2[0]

    overlap_start = max(period1[0],period2[0])
    
    overlap_end = min(period1[1],period2[1])
    
    overlap = max(overlap_end - overlap_start,0)
    min_value = 0.000000001
    return overlap/(p1+p2-overlap_start+min_value),overlap/(p1+min_value),overlap/(p2+min_value)
    

def compare_between_2periods(gt_periods,pd_periods):
    
    recalls = [0]*len(gt_periods)
    
    gt_IOUs = [0]*len(gt_periods)
    
    precisions = [0]*len(pd_periods)
    
    pd_IOUs = [0]*len(pd_periods)
    
    for pd_index, pd_period in enumerate(pd_periods):
        for gt_index, gt_period in enumerate(gt_periods):
            IOU, pd_IOU, gt_IOU = periods_IOU(pd_period,gt_period)
            gt_IOUs[gt_index] += gt_IOU
            pd_IOUs[pd_index] += pd_IOU
            
    recall = sum(gt_IOU>0.1 for gt_IOU in gt_IOUs)/len(gt_IOUs)

    precision = sum(pd_IOU>0.1 for pd_IOU in pd_IOUs)/len(pd_IOUs)
    
    return recall,precision

def compare_between_2periods2(gt_periods,pd_periods):
    
    recalls = [0]*len(gt_periods)
    
    gt_IOUs = [0]*len(gt_periods)
    
    precisions = [0]*len(pd_periods)
    
    pd_IOUs = [0]*len(pd_periods)
    
    for pd_index, pd_period in enumerate(pd_periods):
        for gt_index, gt_period in enumerate(gt_periods):
            IOU, pd_IOU, gt_IOU = periods_IOU(pd_period,gt_period)
            gt_IOUs[gt_index] += gt_IOU
            pd_IOUs[pd_index] += pd_IOU
            
    recall = [sum(gt_IOU>0.1 for gt_IOU in gt_IOUs),len(gt_IOUs)]

    precision = [sum(pd_IOU>0.1 for pd_IOU in pd_IOUs),len(pd_IOUs)]
    
    return recall,precision

def load_and_eval(_exp_name='',gt_vision='v3'):

    data_dir = 'data_gc/videos_test/'
    
    gt_files = sorted(glob.glob(os.path.join(data_dir,"xiehe2111_2205/"+gt_vision+"/*.mp4.txt")))
    
    #exp_name = 'WJ_V1_with_mfp7x-22-2_ppsa_best_roifix'
    #exp_name = 'WJ_V1_with_mfp7x-22-2_best_roifix'
    exp_name = 'WJ_V1_with_mfp7-22-2_v4-0_best_roifix_0.3_vis'
    if _exp_name != '':
        exp_name = _exp_name
    
    print(exp_name)
    
    pd_files = sorted(glob.glob(os.path.join(data_dir,'xiehe2111_2205_'+exp_name+'/*.mp4.txt')))
    
    A_recalls = [0,0.0001]
    
    A_precisions = [0,0.0001]
    
    for gt_file,pd_file in zip(gt_files,pd_files):
        
        #print(os.path.basename(gt_file))
        
        gt_periods = periods_filter(get_positive_periods(gt_file))
        pd_periods = periods_filter(get_positive_periods(pd_file),2)
        
        r,p = compare_between_2periods2(gt_periods,pd_periods)
        
        A_recalls[0] += r[0]
        A_recalls[1] += r[1]
        
        A_precisions[0] += p[0]
        A_precisions[1] += p[1]
        
    #A_recalls/=len(gt_files)
    #A_precisions/=len(pd_files)
        
    print("recall:{0}".format(A_recalls[0]/A_recalls[1]))
    print("precision:{0}".format(A_precisions[0]/A_precisions[1]))

def load_and_eval_list():
    gt_vision = 'v3'
    folders_list = ['WJ_V1_with_mfp7-22-2_retrain_recovery_best_roifix_0.3',
                    'WJ_V1_with_mfp7-22-2_retrain_recovery1_best_roifix_0.3',
                    'WJ_V1_with_mfp7-22-2-0_best_roifix_0.3',
                    'WJ_V1_with_mfp7-22-2-1_best_roifix_0.3',
                    'WJ_V1_with_mfp7-22-2-2_best_roifix_0.3',
                    'WJ_V1_with_mfp7-22-2-3_best_roifix_0.3',
                    'WJ_V1_with_mfp7-22-2-4_best_roifix_0.3',
                    'WJ_V1_with_mfp7-22-2-5_best_roifix_0.3',
                    'WJ_V1_with_mfp7-22-2-recovery_best_roifix_0.3',
                    'WJ_V1_with_mfp7-22-2-recurrent_best_roifix_0.3'
                    ]
    
    folders_list += ['WJ_V1_with_mfp7-22-2-7_best_roifix_0.3',
                    'WJ_V1_with_mfp7-22-2-8_best_roifix_0.3',]
    
    folders_list += ['WJ_V1_with_mfp7-22-2-8-ns_best_roifix_0.3',
                    'WJ_V1_with_mfp7-22-2-9-ns_best_roifix_0.3',]
    folders_list += ['WJ_V1_with_mfp7-22-2-10-ns_best_roifix_0.3',
                     'WJ_V1_with_mfp7-22-2-11_best_roifix_0.3',
                     'WJ_V1_with_mfp7-22-2-12_best_roifix_0.3',
                     'WJ_V1_with_mfp7-22-2-13_best_roifix_0.3',
                     'WJ_V1_with_mfp7-22-2-14_best_roifix_0.3',
                     'WJ_V1_with_mfp7-22-2-18_best_roifix_0.3',
                     'WJ_V1_with_mfp7-22-2-19_best_roifix_0.3',
                     'WJ_V1_with_mfp7-22-2-20_best_roifix_0.3',
                     'WJ_V1_with_mfp7-22-2-21_best_roifix_0.3',
                     'WJ_V1_with_mfp7-22-2-22_best_f2_roifix_0.3',]
    for folder in folders_list:
        load_and_eval(folder,gt_vision)

def generate_eval_images_from_videos():
    data_dir = 'data_gc/videos_test/'
    
    gt_files = sorted(glob.glob(os.path.join(data_dir,"xiehe2111_2205/*.mp4.txt")))            
    
if __name__ == '__main__':
    '''
    gt_periods = get_positive_periods('data_gc/videos_test/xiehe2111_2205/20211123_104815_03_r02_olbs290.mp4.txt')
    
    pd_periods = get_positive_periods('data_gc/videos_test/xiehe2111_2205_WJ_V1_with_mfp3-0_best/20211123_104815_03_r02_olbs290.mp4.txt')
    
    pd_periods = periods_filter(pd_periods,2)
    
    compare_between_2periods(gt_periods,pd_periods)
    '''
    
    #load_and_eval()
    load_and_eval_list()
    
    