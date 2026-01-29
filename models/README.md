# 模型搭建指南
## 📊 一、模型方法全景图
text
```
┌─────────────────────────────────────────────────────┐
│               鸟声识别模型方法演进                   │
├─────────────────────────────────────────────────────┤
│  传统方法        │ 深度学习时代     │ 前沿探索        │
│  • MFCC + SVM    │ • CNN系列        │ • Vision       │
│  • HMM           │ • CRNN           │   Transformer  │
│  • GMM           │ • LSTM/GRU       │ • Audio Mamba  │
│                  │ • ResNet         │ • ConvNeXt     │
│                  │ • EfficientNet   │ • Hybrid       │
│                  │ • MobileNet      │   Models       │
└─────────────────────────────────────────────────────┘
```
## 🔧 二、具体模型方法详解
### 1. 传统机器学习方法（基础，可作对比基线）
| 方法 | 原理 | 优缺点 | 适合度 |
|------|------|--------|--------|
| **MFCC + SVM** | 提取MFCC特征 + 支持向量机分类 | 简单快速，但准确率有限（~70%） | ⭐⭐ |
| **MFCC + Random Forest** | 多棵决策树集成 | 抗过拟合，可解释性强 | ⭐⭐ |
| **HMM（隐马尔可夫）** | 建模声音时序变化 | 适合时序数据，但训练复杂 | ⭐ |
| **GMM（高斯混合模型）** | 对特征分布建模 | 计算快，适合小数据集 | ⭐ |
python
```
# MFCC + SVM示例
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# 提取MFCC特征
mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
mfccs_mean = np.mean(mfccs.T, axis=0)  # 取均值作为特征

# 训练SVM
svm = SVC(kernel='rbf', C=1.0)
svm.fit(X_train, y_train)
accuracy = svm.score(X_test, y_test)
```
### 2. 卷积神经网络（CNN）系列（主流选择）
#### （1）基础CNN
python
```
class BasicCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(128, num_classes)
```
优点：简单易实现，参数少，训练快

缺点：特征提取能力有限

适合度：⭐⭐⭐⭐（入门首选）

#### （2）ResNet（残差网络）
python
```
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)
# 修改第一层（输入通道）和最后一层（输出类别）
resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)
```
优点：解决梯度消失，深层网络稳定

预训练权重：可用ImageNet预训练

适合度：⭐⭐⭐⭐⭐（强烈推荐）

#### （3）EfficientNet（效率最高）
python
```
from efficientnet_pytorch import EfficientNet
# EfficientNet-B0到B7，B0最轻量
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
# 需要修改输入通道数
```
优点：精度和效率的最佳平衡

复合缩放：同时调整深度、宽度、分辨率

适合度：⭐⭐⭐⭐⭐（研究常用）

#### （4）MobileNet系列（专为移动端）
python
```
# MobileNetV2
mobilenet = models.mobilenet_v2(pretrained=True)
mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, num_classes)

# MobileNetV3（更轻量）
from torchvision.models import mobilenet_v3_small
```
优点：极轻量，适合部署

缺点：准确率略低于ResNet/EfficientNet

适合度：⭐⭐⭐⭐（部署必选）

### 3. 循环神经网络（RNN）系列（适合时序）
#### （1）LSTM/GRU
python
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 双向所以×2
    
    def forward(self, x):
        # x: (batch, time_steps, features)
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # 拼接最后时刻的隐状态
        return self.fc(hidden)
优点：捕捉时序依赖

适合：原始音频波形或MFCC序列

适合度：⭐⭐⭐

#### （2）CRNN（CNN + RNN）
python
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # CNN部分提取局部特征
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2)
        )
        # RNN部分捕捉时序
        self.rnn = nn.GRU(128*8, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_classes)
优点：结合CNN的局部感知和RNN的时序建模

适合：梅尔频谱图（时间×频率）

适合度：⭐⭐⭐⭐

### 4. Transformer系列（前沿热点）
#### （1）Vision Transformer (ViT)
python
```
from transformers import ViTModel, ViTConfig

# 将频谱图视为图像
config = ViTConfig(
    image_size=224,  # 频谱图尺寸
    patch_size=16,
    num_channels=1,  # 单通道（灰度图）
    num_classes=num_classes
)
vit = ViTModel(config)
```
优点：全局注意力，强特征提取

缺点：需要大量数据，计算资源大

适合度：⭐⭐⭐（数据量大时推荐）

#### （2）Audio Spectrogram Transformer (AST)
python
```
# 专门为音频设计的Transformer
from transformers import ASTModel

ast = ASTModel.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=num_classes
)
```
优点：音频专用，在AudioSet上预训练

缺点：模型大（~90M参数）

适合度：⭐⭐⭐⭐（如果算力足够）

#### （3）Mamba（最新状态空间模型）
python
```
# Mamba是2024年的新模型，比Transformer更高效
# 目前实现较少，但很有前景
from mamba_ssm import Mamba

class AudioMamba(nn.Module):
    def __init__(self, d_model, n_layers, num_classes):
        super().__init__()
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model) for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(d_model, num_classes)
```
优点：线性时间复杂度，适合长序列

缺点：较新，实现和优化较少

适合度：⭐⭐⭐（有探索价值）

### 5. 轻量化/部署专用模型
| 模型 | 参数量 | 特点 | 适合度 |
|------|--------|------|--------|
| **ShuffleNetV2** | ~2M | 通道混洗，高效 | ⭐⭐⭐⭐ |
| **SqueezeNet** | ~1.2M | Fire模块，极轻量 | ⭐⭐⭐ |
| **GhostNet** | ~3.9M | 幽灵模块，性价比高 | ⭐⭐⭐⭐ |
| **NanoNet** | ~0.5M | 超轻量，适合微控制器 | ⭐⭐⭐ |
## 🎯 三、针对你项目的推荐策略
### 阶段1：建立基线（第1-2周）
python
```
# 每人选择一种不同的CNN作为基线
baseline_models = {
    "成员A": "ResNet18",
    "成员B": "EfficientNet-B0", 
    "成员C": "MobileNetV2",
    "成员D": "SimpleCNN（自定义）"
}
```
目标：比较不同CNN在相同数据上的表现

### 阶段2：引入时序/注意力（第3-4周）
python
```
advanced_models = {
    "CRNN": "CNN + LSTM",
    "CNN+Attention": "CNN + 自注意力",
    "ViT-Small": "小规模Vision Transformer"
}
```
### 阶段3：轻量化优化（第5-6周）
python
```
lightweight_models = {
    "知识蒸馏": "EfficientNet-B2 → MobileNetV2",
    "剪枝+量化": "ResNet18剪枝后量化",
    "神经架构搜索": "自动搜索最优小模型"
}
```
### 阶段4：集成/融合（可选）
python
```
# 模型融合策略
ensemble_methods = {
    "投票融合": "多个模型投票决定",
    "加权平均": "按验证集表现加权",
    "Stacking": "用模型输出训练元分类器"
}
```
