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

def load_and_eval():

    data_dir = 'data_gc/videos_test/'
    
    gt_files = sorted(glob.glob(os.path.join(data_dir,"xiehe2111_2205/*.mp4.txt")))
    
    pd_files = sorted(glob.glob(os.path.join(data_dir,'xiehe2111_2205_WJ_V1_with_mfp7-22-1_best_roifix/*.mp4.txt')))
    
    A_recalls = 0
    
    A_precisions = 0
    
    for gt_file,pd_file in zip(gt_files,pd_files):
        
        print(os.path.basename(gt_file))
        
        gt_periods = periods_filter(get_positive_periods(gt_file))
        pd_periods = periods_filter(get_positive_periods(pd_file),2)
        
        r,p = compare_between_2periods(gt_periods,pd_periods)
        
        A_recalls += r
        
        A_precisions += p
        
    A_recalls/=len(gt_files)
    A_precisions/=len(pd_files)
        
    print(A_recalls)
    print(A_precisions)
        
    
if __name__ == '__main__':
    '''
    gt_periods = get_positive_periods('data_gc/videos_test/xiehe2111_2205/20211123_104815_03_r02_olbs290.mp4.txt')
    
    pd_periods = get_positive_periods('data_gc/videos_test/xiehe2111_2205_WJ_V1_with_mfp3-0_best/20211123_104815_03_r02_olbs290.mp4.txt')
    
    pd_periods = periods_filter(pd_periods,2)
    
    compare_between_2periods(gt_periods,pd_periods)
    '''
    
    load_and_eval()
    
    