***update: v2 model release on 11/14/2022!***

1.检查运行环境中依赖库是否符合requirements.txt中版本要求；

2.安装接口：
``` shell
python setup.py install
```
3.模型参数存放位置（nas服务器）：

v1: /data_us/qilei/胃部病变检测模型，其中single_category.pt用于癌检测，multi_categories.pt用于多病种（溃疡，糜烂，癌等）检测；

***update!***

v2: /data_us/qilei/胃部病变检测模型/binary_categories_y7.pt；该模型输出结果为二分类，0代表癌变（高风险区域），1代表其他类别区域；默认置信度为0.3，可在0.3附近进行置信度调整；

4.接口参数说明请参考yolov7gastroscopy/inference.py，用例请参考test_inference.py