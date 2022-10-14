1.检查运行环境中依赖库是否符合requirements.txt中版本要求；

2.安装接口：
``` shell
python setup.py install
```
3.模型参数存放位置：

    /data_us/qilei/胃部病变检测模型，其中single_category.pt用于癌检测，multi_categories.pt用于多病种（溃疡，糜烂，癌等）检测；

3.用例请参考test_inference.py