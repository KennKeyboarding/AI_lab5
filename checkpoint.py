from preprocess import *

# 调用 main() 函数以加载数据
train_images, train_texts, train_labels, test_images, test_texts = main()

train_data = pd.read_csv('train.txt')
test_data = pd.read_csv('test_without_label.txt')

# 将 test_data 的 guid 列转换为整数（但好像没用？）
test_data['guid'] = test_data['guid'].astype(int)

print("Training Samples:")
for i in range(5):  
    print(f"Sample {i+1}:")
    print(f"  GUID: {train_data.iloc[i]['guid']}")
    print(f"  Text: {train_texts[i][:50]}...")  # 打印前 50 个字符
    print(f"  Label: {train_labels[i]}")
    print("-" * 50)

print("Test Samples:")
for i in range(5):  
    print(f"Sample {i+1}:")
    print(f"  GUID: {test_data.iloc[i]['guid']}")
    print(f"  Text: {test_texts[i][:50]}...") 
    print("-" * 50)