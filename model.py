import torch
import torch.nn as nn
from transformers import BertModel
from torchvision.models import resnet50

class MultimodalModel(nn.Module):
    def __init__(self, text_model_name='bert-base-uncased', num_classes=3):
        super(MultimodalModel, self).__init__()
        # 文本特征提取器（BERT）
        self.text_model = BertModel.from_pretrained(text_model_name)
        
        # 图像特征提取器（ResNet）
        # 加载 ResNet-50 模型，但不加载预训练权重
        # self.image_model = resnet50(pretrained=False)
        self.image_model = resnet50(weights=None)
        # 从本地加载预训练权重文件
        self.image_model.load_state_dict(torch.load('resnet50-0676ba61.pth'))
        self.image_model.fc = nn.Identity()  # 去掉 ResNet 的最后一层全连接层
        
        # 文本特征维度（BERT 的 hidden_size）
        text_feature_dim = self.text_model.config.hidden_size
        # 图像特征维度（ResNet 的输出维度）
        image_feature_dim = 2048  # ResNet-50 的输出维度是 2048
        
        # 分类器
        self.fc = nn.Linear(text_feature_dim + image_feature_dim, num_classes)
    
    def forward(self, text_input_ids, text_attention_mask, image):
        # 提取文本特征
        text_features = self.text_model(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        ).last_hidden_state[:, 0, :]  # 取 [CLS] 标记的输出
        
        # 提取图像特征
        image_features = self.image_model(image)
        
        # 拼接文本和图像特征
        combined_features = torch.cat((text_features, image_features), dim=1)
        
        # 分类
        output = self.fc(combined_features)
        return output