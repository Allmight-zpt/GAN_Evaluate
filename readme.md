# 实验室工作 GAN-Evaluate
## 项目介绍
从以下三个方面对GAN模型生成结果进行评估：
- 生成比例 
  - 基于分类模型 
  - 基于孪生网络
- 生成质量
  - 基于FID
- 生成多样性

## 目录介绍
- data: 数据
- evaluate: 评估方法
- pretrain_weights: 预训练模型
- result: 实验结果
- scripts: 运行脚本
- utils: 实用工具

## 运行
模型训练过程中生成图片的FID变化
```shell
python ./evaluate/FID_In_OOD_Task.py --weightDir YourTaskName
```
单个模型生成图片的FID
```shell
python ./evaluate/Generate_Quality_FID.py
```
单个模型生成图片的分类结果
```shell
python ./evaluate/Generate_Proportion_Classify.py
```
单个模型生成图片基于孪生网络的相似度计算结果
```shell
python ./evaluate/Generate_Proportion_Siamese.py
```

## 注意
1. 运行 FID_In_OOD_Task.py 前先将OOD数据训练过程中保存的所有模型权重放到 pretrain_weights\GANs\YourTaskName 目录中
2. result 文件命名规则：模型 + ID数据采样类型 + (OOD数据集名称) + (OOD数据采样量) + (OOD数据采样类型) + (是否使用ID数据集进行预训练)
3. 上面的命名规则只说明了来源与GAN_OOD项目中的FedGAN_OC_Add_OOD_Data.py文件的结果的命名，其他文件的还未说明！！！！！！！

## 代办
把剩下三个evaluate改成命令行形式读取参数

1. 使用一个100000epoch的预训练模型，包括基于Mnist FashionMnist两种数据集的预训练模型
2. 计算两个Fid的相关性  和num与Fid的相关性


## 相关性分析结果
### 说明
- Corr_Fids: 表示生成的OOD数据(FashionMnist)的Fid和生成的ID数据(Mnist)的Fid之间的相关性
- Corr_Mnist_Fid_Num: 表示生成的ID数据(Mnist)的Fid和生成的数据量之间的相关性
- Corr_FashionMnist_Fid_Num: 表示生成的OOD数据(Mnist)的Fid和生成的数据量之间的相关性
- 纵轴表示相关性，横轴表示不同实验，实验命名规则: (OOD数据集名称) + (OOD数据采样量) + (OOD数据采样类型) + (是否使用ID数据集进行预训练)，以FM_100_IID_n为例，表示FM是OOD数据集，总共采集100个OOD数据，每个节点得到的OOD数据是IID的，不使用ID数据预训练模型(即OOD数据和ID数据同时训练)。
### pearson correlation
![pearson_corr.png](result%2FCorrelation%2Ffigs%2Fpearson_corr.png)
### spearman correlation
![spearman_corr.png](result%2FCorrelation%2Ffigs%2Fspearman_corr.png)
### kendall correlation
![kendall_corr.png](result%2FCorrelation%2Ffigs%2Fkendall_corr.png)