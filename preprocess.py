import os
import pandas as pd
from PIL import Image
from torchvision import transforms
import re
import string
import glob

# 图像预处理
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 清理文本中的噪声，包括 @用户名、http 链接、# 标签、RT、标点符号等
def clean_text(text):
    text = re.sub(r'\bRT\b', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'#(\w+)', r'\1', text)   # 将 # 标签转换为普通单词
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 读取文本文件（自动检测编码）
def read_text_file(file_path):
    encodings = ['utf-8', 'gbk', 'latin-1']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read().strip()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Failed to decode {file_path} with encodings: {encodings}")

# guid 查找文件路径 
def find_file_path(data_dir, guid, extension):
    matches = glob.glob(os.path.join(data_dir, f'{guid}*{extension}'))
    if matches:
        return matches[0]  # 返回第一个匹配的文件
    return None

# 加载图像和文本数据，并清理文本
def load_data(data, data_dir):
    images = []
    texts = []
    labels = []
    for guid, tag in data.values:
        guid = int(float(guid))  
        # 查找图像文件
        image_path = find_file_path(data_dir, guid, '.jpg')  
        if not image_path:
            print(f"Warning: Image for guid {guid} not found. Skipping.")
            continue
        
        image = Image.open(image_path).convert('RGB')
        image = image_transform(image)
        images.append(image)
        
        # 查找文本文件
        text_path = find_file_path(data_dir, guid, '.txt')  
        if not text_path:
            print(f"Warning: Text for guid {guid} not found. Skipping.")
            continue
        
        # 加载文本并清理
        try:
            text = read_text_file(text_path)
            text = clean_text(text)
            texts.append(text)
        except UnicodeDecodeError as e:
            print(f"Warning: Failed to decode {text_path}. Skipping. Error: {e}")
            continue
        
        if 'tag' in data.columns:
            labels.append(tag)
    
    return images, texts, labels if labels else None

# 主函数
def main():
    train_data = pd.read_csv('train.txt')
    test_data = pd.read_csv('test_without_label.txt')

    train_images, train_texts, train_labels = load_data(train_data, 'data')
    test_images, test_texts, _ = load_data(test_data, 'data')

    print(f"Loaded {len(train_images)} training samples.")
    print(f"Loaded {len(test_images)} test samples.")

    return train_images, train_texts, train_labels, test_images, test_texts

if __name__ == '__main__':
    train_images, train_texts, train_labels, test_images, test_texts = main()