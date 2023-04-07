#python test.py --weights /data1/qilei_chen/DEVELOPMENTS/yolov7/out/WJ_V1_with_mfp1/yolov7-wj_v1_with_fp/weights/best.pt --data data/wj_v1_with_fp.yaml --batch-size 32 --device 0 --exist-ok

#python test.py --weights /data1/qilei_chen/DEVELOPMENTS/yolov7/out/WJ_V1_with_mfp2/yolov7-wj_v1_with_fp/weights/best.pt --data data/wj_v1_with_fp.yaml --batch-size 32 --device 0 --exist-ok

#python test.py --weights out/WJ_V1_with_mfp3-1/yolov7-wj_v1_with_fp/weights/best.pt --data data/wj_v1_with_fp.yaml --batch-size 32 --device 0 --exist-ok --verbose

python test_org_dev.py \
    --weights out/WJ_V1_with_mfp7-22-2_retrain/yolov7-wj_v1_with_fp/weights/best.pt \
    --data data/wj_v1_with_fp.yaml \
    --batch-size 32 \
    --exist-ok \
    --device 0 \
    --project runs/WJ_V1_with_mfp7-22-2_retrain \
    --save-json