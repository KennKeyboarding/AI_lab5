import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model import MultimodalModel
from preprocess import main
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultimodalModel().to(device)
model.load_state_dict(torch.load('best_model.pth'))  # 最佳模型
model.eval()

test_data = pd.read_csv('test_without_label.txt')
_, _, _, test_images, test_texts = main()  # 只接收测试集的图像和文本数据

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, images, texts, tokenizer, max_length=128):
        self.images = images
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        image = self.images[idx]
        
        # 对文本进行编码
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'image': image,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

test_dataset = MultimodalDataset(test_images, test_texts, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 预测函数
def predict(model, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            
            # 前向传播
            outputs = model(input_ids, attention_mask, images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
    return predictions

predictions = predict(model, test_loader)

# 将预测结果映射为标签
label_map = {0: 'positive', 1: 'neutral', 2: 'negative'}
test_data['tag'] = [label_map[pred] for pred in predictions]

test_data.to_csv('test_with_label.txt', index=False)
print("预测结果已保存到 test_with_label.txt")