import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary

from cotracker.predictor import CoTrackerPredictor
import os
import torch.nn.functional as F
#
# from test import StudentEncoder


class UniformTransformedDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir: COCO 数据集的根目录，例如 'D:/Code/co-tracker/coco/val2017/val2017'
        """
        self.root_dir = root_dir
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.interp_shape = (384, 512)
        self.window_len = 8

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_paths = self.files[index:index + self.window_len]  # 取出连续的8张图片路径
        video_frames = []
        for img_path in img_paths:
            # 以BGR模式读取图片
            image_bgr = cv2.imread(img_path)
            # 将BGR图片转换为RGB图片
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            # 将图片转换为Tensor
            frame_tensor = torch.from_numpy(image_rgb.transpose((2, 0, 1))).float().unsqueeze(0)
            frame_tensor = F.interpolate(frame_tensor, tuple(self.interp_shape), mode="bilinear", align_corners=True)  # 插值调整大小
            video_frames.append(frame_tensor)

        video = torch.cat(video_frames, dim=0).unsqueeze(0)  # 拼接成视频张量
        B, T, C, H, W = video.shape
        video = video.reshape(B * T, C, H, W)
        video = F.interpolate(video, tuple(self.interp_shape), mode="bilinear", align_corners=True)
        video = video.reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])
        B, T, C, H, W = video.shape
        S = self.window_len
        video = 2 * (video / 255.0) - 1.0
        pad = (S - T % S) % S
        video = F.pad(video.reshape(B, 1, T, C * H * W), (0, 0, 0, pad), "replicate").reshape(
            B, -1, C, H, W
        )
        video = video.reshape(-1, C, H, W)
        return video





# 设置设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


# 定义训练和评估函数
def train_model(teacher, student, train_loader, test_loader, epochs=10, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(student.parameters(), lr=lr)
    best_loss = float('inf')
    teacher.eval()  # 确保老师模型处于评估模式

    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        for batch_idx, inputs in enumerate(train_loader):
            inputs = inputs.to(device).squeeze()
            with torch.no_grad():
                teacher_outputs = teacher(inputs)

            optimizer.zero_grad()
            student_outputs = student(inputs)
            loss = criterion(student_outputs, teacher_outputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # 实时显示loss
            if (batch_idx + 1) % 10 == 0:  # 每10个batch打印一次
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item()}")

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

        # 每个epoch结束时评估模型
        val_loss = evaluate_model(teacher, student, test_loader)
        print(f'Epoch {epoch + 1}: Val Loss: {val_loss}')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(student.state_dict(), 'best_model.pth')
            print("Saved Best Model")


def evaluate_model(teacher, student, loader):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for inputs in loader:
            inputs = inputs.to(device).squeeze()
            with torch.no_grad():
                teacher_outputs = teacher(inputs)
            student_outputs = student(inputs)
            loss = criterion(student_outputs, teacher_outputs)
            total_loss += loss.item()
    print(f"Test Loss: {total_loss / len(loader)}")
    return total_loss / len(loader)


# 实例化 COCO 数据集
val_dataset = UniformTransformedDataset('D:/Code/co-tracker/coco/val2017/val2017/val')
train_dataset = UniformTransformedDataset('D:/Code/co-tracker/coco/val2017/val2017/train')
# 创建 DataLoader，设置 batch_size 为 1
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

model = CoTrackerPredictor("D:\Code\co-tracker\checkpoints\cotracker2.pth").to(device)
# 实例化老师和学生模型
teacher = model.model.fnet
summary(teacher, (8, 3, 384, 512))
# student = StudentEncoder().to(device)

# 开始训练
# train_model(teacher, student, train_dataloader, val_dataloader, epochs=120, lr=0.001)
