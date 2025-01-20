import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from model import MultimodalModel
from preprocess import main

# 超参数
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5

# 标签映射字典
label_map = {
    'positive': 0,
    'neutral': 1,
    'negative': 2
}

train_images, train_texts, train_labels, test_images, test_texts = main()

# 将字符串标签映射为整数标签
train_labels = [label_map[label] for label in train_labels]

# 划分训练集和验证集
train_images, val_images, train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_images, train_texts, train_labels, test_size=0.2, random_state=42
)

# 初始化 BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 自定义数据集类
class MultimodalDataset(Dataset):
    def __init__(self, images, texts, labels, tokenizer, max_length=128):
        self.images = images
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        image = self.images[idx]
        label = self.labels[idx]
        
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
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)  # 标签已经是整数，直接转换为张量
        }

# 创建数据集和数据加载器
train_dataset = MultimodalDataset(train_images, train_texts, train_labels, tokenizer)
val_dataset = MultimodalDataset(val_images, val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultimodalModel().to(device)
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# 训练和验证函数
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS):
    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            outputs = model(input_ids, attention_mask, images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}')
        
        # 验证
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                # 前向传播
                outputs = model(input_ids, attention_mask, images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # 计算准确率
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels).item()
        
        print(f'Validation Loss: {val_loss/len(val_loader)}, Accuracy: {correct/len(val_dataset)}')

train_and_validate(model, train_loader, val_loader, criterion, optimizer)

torch.save(model.state_dict(), 'multimodal_model.pth')