import torch
import torch.nn as nn
from transformers import BertModel
from torchvision.models import resnet50

class MultimodalModel(nn.Module):
    def __init__(self, text_model_name='bert-base-uncased', num_classes=3, dropout_prob=0.5):
        super(MultimodalModel, self).__init__()
        # BERT
        self.text_model = BertModel.from_pretrained(text_model_name)
        
        # ResNet
        self.image_model = resnet50(weights=None)
        self.image_model.load_state_dict(torch.load('resnet50-0676ba61.pth'))
        self.image_model.fc = nn.Identity()  # 去掉 ResNet 的最后一层全连接层
        
        text_feature_dim = self.text_model.config.hidden_size
        image_feature_dim = 2048 
        
        self.dropout = nn.Dropout(dropout_prob)
        
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

        combined_features = self.dropout(combined_features)
        
        output = self.fc(combined_features)
        return output