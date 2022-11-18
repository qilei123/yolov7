#python test.py --weights /data1/qilei_chen/DEVELOPMENTS/yolov7/out/WJ_V1_with_mfp1/yolov7-wj_v1_with_fp/weights/best.pt --data data/wj_v1_with_fp.yaml --batch-size 32 --device 0 --exist-ok

#python test.py --weights /data1/qilei_chen/DEVELOPMENTS/yolov7/out/WJ_V1_with_mfp2/yolov7-wj_v1_with_fp/weights/best.pt --data data/wj_v1_with_fp.yaml --batch-size 32 --device 0 --exist-ok

python evaluation.py --weights out/WJ_V1_with_mfp3-1/yolov7-wj_v1_with_fp/weights/best.pt --data data/wj_v1_with_fp.yaml --batch-size 32 --device 1 --exist-ok --verbose --save-txt --save-conf --iou-thres 0.0 --conf-thres 0.3