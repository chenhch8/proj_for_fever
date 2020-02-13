# 环境
```
transformers
pytorch=1.1.0
torchvision
tqdm
numpy
tensorboardX
tensorboard
```

# 训练
1. 需指定参数train_data，可选为
- train_process(5)-v1.jsonl: 未对候选文档集进行约束
- train_process(5)-v2.jsonl: 约束候选文档集
- train_process(5)-v3,jsonl: 约束候选文档集+指代消解
2. 需指定参数data_loader，可选为
- v1: 通过采样方式构造batch
- v3: 先构造batch，再进行采样

# 数据链接
待上传
