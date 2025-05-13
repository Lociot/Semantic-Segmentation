import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
df = pd.read_csv('bieudo.csv')

# Loại bỏ dấu cách thừa trong các tên cột nếu có
df.columns = df.columns.str.strip()

# Tạo biểu đồ
fig, axes = plt.subplots(3, 1, figsize=(10, 9))

# Biểu đồ cho Loss
axes[0].plot(df['Epoch'], df['Train Loss'], label='Train Loss', color='b', marker='o')
axes[0].plot(df['Epoch'], df['Validation Loss'], label='Validation Loss', color='g', marker='x')
axes[0].set_title('Loss Over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

# Biểu đồ cho IoU
axes[1].plot(df['Epoch'], df['Train IoU'], label='Train IoU', color='b', marker='o')
axes[1].plot(df['Epoch'], df['Validation IoU'], label='Validation IoU', color='g', marker='x')
axes[1].set_title('IoU Over Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('IoU')
axes[1].legend()

# Biểu đồ cho Accuracy
axes[2].plot(df['Epoch'], df['Train Accuracy'], label='Train Accuracy', color='b', marker='o')
axes[2].plot(df['Epoch'], df['Validation Accuracy'], label='Validation Accuracy', color='g', marker='x')
axes[2].set_title('Accuracy Over Epochs')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Accuracy')
axes[2].legend()

# Hiển thị các biểu đồ
plt.tight_layout()
plt.show()
