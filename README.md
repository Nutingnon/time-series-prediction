
# 安装指南
pip install numpy matplotlib pandas scikit-learn torch==1.11.0

#  dataset
```train_dev1.csv``` :: 公司提供的完整数据集

```train_dev_8.csv``` :: 完整数据集下 pool_id=8 对应部分的数据

```train_dev_DF.csv``` :: 针对 train_dev1 进行数据分析得到的结果

# 启动模型
```
cd PatchTST-main/MultiPatch_super/scripts/MultiPatch
bash train_dev1.sh
```

# 添加新的数据集

## 操作步骤
1. 上传数据集CSV文件 :: [./dataset]目录下

2. 添加数据集名称 :: [./data_provider/data_factory.py] 文件中第4行 ```data_dict``` 变量内添加 ```'your_dataName': Dataset_Train_dev,```

3. 修改模型脚本的参数 :: [./scripts/LSTM/train_dev1.sh]  [./scripts/MultiPatch/train_dev1.sh] 

4. 运行脚本 :: ```bash train_dev1.sh```

5. 查看结果 :: 脚本同一级目录下 logs  test_results  result.txt

## 模型参数和配置
1. 查找参数定义 :: [./run_longExp.py]文件

2. 修改参数 :: 在具体的脚本中修改参数 [./scripts/LSTM/train_dev1.sh]  [./scripts/MultiPatch/train_dev1.sh]

3. 常用参数 ::
    model_name  模型名称
    data_path_name  数据集名称
    seq_len  输入序列的长度
    pred_len  预测序列的长度
    e_layers  编码器层数
    train_epochs  训练轮数
    batch_size  批量大小
    n_heads  需要预测的列数
    enc_in  通道数 
    learning_rate 优化器学习率