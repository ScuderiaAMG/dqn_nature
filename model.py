# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class DQN(nn.Module):
#     def __init__(self, input_shape, num_actions):
#         super(DQN, self).__init__()
#         self.input_shape = input_shape
#         self.num_actions = num_actions

#         # 原论文的卷积结构（保持不变）
#         self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

#         self.fc1 = nn.Linear(self.feature_size(), 512)
#         self.fc2 = nn.Linear(512, num_actions)

#     def feature_size(self):
#         return self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.input_shape)))).view(1, -1).size(1)

#     def forward(self, x):
#         x = x.float() / 255.0  # 归一化（原代码可能已有，确保保留）
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)

# # 新增Huber损失函数
# def huber_loss(x, delta=1.0):
#     """Huber损失：误差小时用MSE，误差大时用L1"""
#     return torch.where(
#         x.abs() <= delta,
#         0.5 * x.pow(2),
#         delta * (x.abs() - 0.5 * delta)
#     ).mean()
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.input_shape = input_shape  # (4,84,84)
        self.num_actions = num_actions

        # 论文原版卷积结构（恢复512全连接层，适配RTX 4090）
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(self.feature_size(), 512)  # 恢复为512单元
        self.fc2 = nn.Linear(512, num_actions)

    def feature_size(self):
        """计算卷积层输出的特征维度"""
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.input_shape)))).view(1, -1).size(1)

    def forward(self, x):
        """前向传播（AMP会自动转为FP16计算）"""
        x = x.float() / 255.0  # 归一化到[0,1]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def huber_loss(x, delta=1.0):
    """Huber损失：平衡MSE和L1损失，提升稳定性"""
    return torch.where(
        x.abs() <= delta,
        0.5 * x.pow(2),
        delta * (x.abs() - 0.5 * delta)
    ).mean()