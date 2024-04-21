import torch.nn as nn



class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="instance", stride=1):
        super(ResidualBlock, self).__init__()

        # 保持卷积核大小和填充，修改步长和通道数来调整输出尺寸和复杂度
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

        if norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
        else:
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()

        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                nn.InstanceNorm2d(planes) if norm_fn == "instance" else nn.Sequential()
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class StudentEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=128, norm_fn="instance"):
        super(StudentEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, output_dim // 4, kernel_size=7, stride=1, padding=3)
        self.norm1 = nn.InstanceNorm2d(output_dim // 4) if norm_fn == "instance" else nn.Sequential()
        self.relu1 = nn.ReLU(inplace=True)

        # 增加残差块，并调整步长以减少尺寸下降
        self.layer1 = ResidualBlock(output_dim // 4, output_dim // 2, norm_fn, stride=2)
        self.layer2 = ResidualBlock(output_dim // 2, output_dim // 2, norm_fn, stride=2)
        self.layer3 = ResidualBlock(output_dim // 2, output_dim, norm_fn, stride=2)

        # 上采样层，通过插值方法调整到所需尺寸
        self.upsample = nn.Upsample(size=(96, 128), mode='bilinear', align_corners=True)

        # 最终调整层，调整通道数和精细化尺寸
        self.adjust_conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.upsample(x)  # 使用上采样来匹配预期的尺寸
        x = self.adjust_conv(x)
        return x


# 模型初始化和应用示例
# model = StudentEncoder()
# sample_input = torch.randn(8, 3, 384, 512)  # 输入张量
# output = model(sample_input)
# print(output.shape)  # 输出维度应为[1, 128, 96, 128]

# 初始化模型
# model = MobileNetV3Encoder()
# model = CoTrackerPredictor("/home/llm/code/co-tracker/checkpoints/cotracker2.pth")
# 实例化老师和学生模型
# teacher = model.model.fnet
# student = StudentEncoder()
# 假设输入
# input_tensor = torch.randn(1, 3, 384, 512)
# output = teacher(input_tensor)
# print(output.shape)
# summary(teacher, (8, 3, 384, 512))#


# /home/llm/.cache/torch/hub/checkpoints/mobilenet_v3_large-8738ca79.pth
# input: X, 3, 384, 512
# output: 1, X, 128, 96, 128
# 初始化模型
