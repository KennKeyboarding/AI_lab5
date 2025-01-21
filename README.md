# 当代人工智能实验五——多模态情感分析
## 10225501419 吴智轩

本次实验的任务是给定配对的文本和图像，预测对应的情感标签。这是一个三分类任务：**positive, neutral, negative**。


## 环境依赖
环境依赖列在 `requirements.txt` 中
可以依次安装，也可以运行
```shell
pip install -r requirements.txt
```
如果使用GPU，需要安装对应版本的Pytorch和CUDA。

使用的主要库：
```
PyTorch：深度学习框架，用于构建和训练模型。
TorchVision：提供图像处理和预训练模型（ResNet）。
Transformers：提供 BERT 模型和分词器。
Scikit-learn：用于数据集划分。
PIL：图像处理。
Pandas：数据处理。
Matplotlib：绘制 Loss 和 Accuracy 曲线。
```

## 文件结构
```
|-- _pycache_
|-- data                  文本和图像数据
    |-- guid.jpg
    |-- guid.txt 
|-- preprocess.py         数据预处理
|-- checkpoint.py         检查清理后的文本
|-- test_gpu.py           检测GPU
|-- model.py              定义模型架构
|-- train.py              训练模型
|-- train_final.py        训练优化
|-- predict.py            测试集预测
|-- ablation.py           消融实验
|-- resnet50-0676ba61.pth 预训练的ResNet50模型
|-- multimodal_model.pth  多模态融合模型权重文件
|-- final_model.pth       最佳模型权重文件
|-- train.txt             数据的guid和对应情感标签
|-- test_without_label.txt  测试集数据guid和空的情感标签
|-- test_with_label.txt   模型预测的测试集数据
|-- Figure1.png           训练和验证loss和accuracy图像
|-- requirements.txt      环境依赖
|-- README.md
```


## 运行代码
进入文件根目录并运行。

需要提前下载预训练模型 ResNet50 ，保存在根目录下。

访问以下链接：
https://download.pytorch.org/models/resnet50-0676ba61.pth

```
python preprocess.py   # 预处理
python train_final.py  # 训练模型
python predict.py      # 测试集预测
python ablation.py     # 消融实验
```


## Reference
https://github.com/RecklessRonan/GloGNN/blob/master/readme.md
https://zhuanlan.zhihu.com/p/719361337?utm_id=0
https://blog.csdn.net/qq_45649076/article/details/120494328
https://blog.csdn.net/ty154542607/article/details/144026744
https://leihuo.163.com/institution/news/20230321/38426_1079015.html