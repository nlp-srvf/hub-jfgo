# 三元组损失文本匹配模型

基于三元组损失(Triplet Loss)的中文文本匹配模型，适用于电信客服意图识别等文本分类任务。

## 项目结构

```
├── config.py              # 配置文件
├── device_utils.py        # 设备工具类，自动选择GPU/CPU
├── chars.txt             # 字符词汇表
├── data/                 # 数据目录
│   ├── schema.json       # 标签映射
│   ├── train.json        # 训练数据
│   ├── valid.json        # 验证数据
│   └── data.json        # 完整数据
├── loader.py             # 数据加载器，处理三元组生成
├── model.py              # 模型定义，基于LSTM的双塔网络
├── evaluate.py           # 评估函数
├── main.py               # 训练脚本
├── predict.py            # 预测脚本
├── run.py                # 运行脚本
└── README.md             # 说明文档
```

## 环境要求

```bash
torch>=1.8.0
numpy>=1.19.0
scikit-learn>=0.24.0
```

## 快速开始

### 1. 训练模型

```bash
python run.py train
```

或者直接运行：

```bash
python main.py
```

### 2. 测试模型架构

```bash
python run.py test
```

### 3. 预测模式

```bash
python run.py predict
```

## 详细说明

### 模型架构

- **编码器**: 双向LSTM + Dropout
- **损失函数**: 三元组损失 (Triplet Loss)
- **输入处理**: 字符级别的文本编码
- **相似度计算**: 余弦相似度和欧氏距离

### 数据格式

训练数据格式为JSON Lines，每行包含：

```json
{
    "questions": ["查询话费", "查话费"],
    "target": "话费查询"
}
```

### 配置参数

主要配置项在 `config.py` 中：

```python
Config = {
    "model_path": "model_output",        # 模型保存路径
    "max_length": 20,                   # 最大文本长度
    "hidden_size": 128,                 # 隐藏层大小
    "epoch": 10,                        # 训练轮数
    "batch_size": 32,                   # 批次大小
    "learning_rate": 1e-3,              # 学习率
}
```

## 核心特性

### 1. 三元组数据生成

自动从训练数据生成三元组：
- **Anchor**: 锚点样本
- **Positive**: 同类别的其他样本  
- **Negative**: 不同类别的样本

### 2. 多设备支持

自动检测和选择最佳设备：
- CUDA GPU
- MPS (Apple Silicon)
- NPU (华为昇腾)
- CPU

### 3. 评估指标

- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数
- Top-3准确率

## 预测接口

### 单文本预测

```python
from predict import TextMatchingPredictor

predictor = TextMatchingPredictor('model_output/best_model.pth', Config)
predictions = predictor.predict("查话费", return_top_k=3)
```

### 批量预测

```python
texts = ["查话费", "修改密码", "开通来电显示"]
predictions = predictor.batch_predict(texts)
```

## 训练过程

1. **数据加载**: 自动生成三元组数据
2. **模型训练**: 使用三元组损失优化
3. **验证评估**: 每个epoch后评估性能
4. **模型保存**: 保存最佳F1分数的模型

## 性能优化

- **梯度裁剪**: 防止梯度爆炸
- **学习率调度**: StepLR动态调整
- **早停机制**: 保存最佳模型
- **数据并行**: 支持多GPU训练

## 示例输出

训练过程：
```
Epoch 1 [0/150 (0.0%)] Loss: 0.823456
Epoch 1 completed in 12.34s, Average Loss: 0.789123
学习率: 0.001000

验证模型...
Validation Set Results:
==================================================
Accuracy:  0.8567
Precision: 0.8543
Recall:    0.8567
F1-Score:  0.8555
==================================================
```

预测结果：
```
输入: 查一下我的话费
预测: 话费查询 (相似度: 0.8934)

Top-3结果:
1. 话费查询 (相似度: 0.8934)
2. 套餐余量查询 (相似度: 0.7562)  
3. 月返费查询 (相似度: 0.6341)
```

## 注意事项

1. **初学者友好**: 代码结构清晰，注释详细
2. **可扩展性**: 模块化设计，易于修改
3. **资源占用**: 模型较轻量，适合CPU训练
4. **中文支持**: 基于字符级编码，支持中文

## 故障排除

1. **CUDA内存不足**: 减小batch_size
2. **训练缓慢**: 检查设备选择和优化设置
3. **过拟合**: 增加dropout或减少模型复杂度