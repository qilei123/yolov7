#train gc
python train_org_dev.py --workers 64 \
    --device 2,3 \
    --batch-size 64 \
    --data data/wj_v1_with_fp3.yaml \
    --cfg cfg/training/yolov7-wj_v1_with_fp3.yaml \
    --weights 'yolov7.pt' \
    --name yolov7-wj_v1_with_fp \
    --hyp data/hyp.scratch.gc.yaml \
    --project 27_yolov7_output/WJ_V1_with_mfp7-22-2-16 \
    --exist-ok \
    --c_criteria \
    --epochs 300 --shuffle_data_epoch
#     --img-size 800
#train ec
# python train_org_dev.py --workers 64 \
#     --device 3 \
#     --batch-size 32 \
#     --data data/ec_v1.yaml \
#     --cfg cfg/training/yolov7-ec_v1.yaml \
#     --weights 'yolov7.pt' \
#     --name yolov7-ec_v1 \
#     --hyp data/hyp.scratch.ec.yaml \
#     --project 27_yolov7_output/EC_V1-2 \
#     --exist-ok \
#     --c_criteria \
#     --epochs 150
    #--img-size 800

#python train.py --workers 8 --device 1 --batch-size 8 --data data/wj_v1.yaml --img 640 640 --cfg cfg/training/yolov7-wj_v1.yaml --weights 'yolov7.pt' --name yolov7-wj_v1 --hyp data/hyp.scratch.custom.yaml --project /data/qilei/DATASETS/WJ_V1/yolov7_single_cls_2 --exist-ok --single-cls --epochs 110

#python train.py --workers 8 --device 1 --batch-size 8 --data data/wj_v1_roi.yaml --img 640 640 --cfg cfg/training/yolov7-wj_v1_roi.yaml --weights 'yolov7.pt' --name yolov7-wj_v1 --hyp data/hyp.scratch.custom.yaml --project /data/qilei/DATASETS/WJ_V1/yolov7_roi --exist-ok --epochs 110

#python train.py --workers 8 --device 1 --batch-size 8 --data data/wj_v1.yaml --img 640 640 --cfg cfg/training/yolov7x-wj_v1.yaml --weights 'yolov7x.pt' --name yolov7x-wj_v1 --hyp data/hyp.scratch.custom.yaml --project /data/qilei/DATASETS/WJ_V1/yolov7_single_cls_2 --exist-ok --single-cls --epochs 110