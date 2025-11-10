# import numpy as np
# import torch
# from collections import namedtuple

# Experience = namedtuple("Experience", ("state", "action", "reward", "next_state", "done"))

# class PrioritizedReplayBuffer:
#     def __init__(self, capacity, alpha=0.6):
#         self.capacity = capacity
#         self.alpha = alpha  # 优先级指数（0=均匀采样，1=完全按优先级）
#         self.memory = []
#         self.priorities = np.zeros((capacity,), dtype=np.float32)
#         self.pos = 0

#     def push(self, *args):
#         """添加经验并设置初始优先级"""
#         max_prio = self.priorities.max() if self.memory else 1.0
#         if len(self.memory) < self.capacity:
#             self.memory.append(Experience(*args))
#         else:
#             self.memory[self.pos] = Experience(*args)
#         self.priorities[self.pos] = max_prio  # 新经验用最大优先级初始化
#         self.pos = (self.pos + 1) % self.capacity

#     def sample(self, batch_size, beta=0.4):
#         """按优先级采样并计算重要性采样权重"""
#         if len(self.memory) == self.capacity:
#             prios = self.priorities
#         else:
#             prios = self.priorities[:self.pos]
        
#         # 计算采样概率（优先级^alpha）
#         probs = prios ** self.alpha
#         probs /= probs.sum()

#         # 采样索引
#         indices = np.random.choice(len(self.memory), batch_size, p=probs)
#         experiences = [self.memory[idx] for idx in indices]

#         # 计算重要性采样权重（IS权重）
#         total = len(self.memory)
#         weights = (total * probs[indices]) **(-beta)
#         weights /= weights.max()  # 归一化，避免权重过大
#         weights = torch.tensor(weights, dtype=torch.float32)

#         # 转换为批量张量
#         batch = Experience(*zip(*experiences))
#         states = torch.stack(batch.state)
#         actions = torch.tensor(batch.action, dtype=torch.long)
#         rewards = torch.tensor(batch.reward, dtype=torch.float32)
#         next_states = torch.stack(batch.next_state)
#         dones = torch.tensor(batch.done, dtype=torch.float32)

#         return (states, actions, rewards, next_states, dones, indices, weights)

#     def update_priorities(self, indices, td_errors):
#         """更新采样经验的优先级（基于TD误差）"""
#         for idx, error in zip(indices, td_errors):
#             self.priorities[idx] = abs(error.item()) + 1e-6  # 避免优先级为0

#     def __len__(self):
#         return len(self.memory)
import numpy as np
import torch
from collections import namedtuple

Experience = namedtuple("Experience", ("state", "action", "reward", "next_state", "done"))

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数（0=均匀采样，1=完全按优先级）
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, *args):
        """存储经验到CPU（避免占用GPU显存）"""
        max_prio = self.priorities.max() if self.memory else 1.0  # 新经验用最大优先级
        if len(self.memory) < self.capacity:
            self.memory.append(Experience(*args))
        else:
            self.memory[self.pos] = Experience(*args)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """按优先级采样，返回CPU张量（后续手动移至GPU）"""
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        # 计算采样概率（优先级^alpha）
        probs = prios ** self.alpha
        probs /= probs.sum()

        # 采样索引和经验
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        experiences = [self.memory[idx] for idx in indices]

        # 计算重要性采样权重（抵消优先级偏差）
        total = len(self.memory)
        weights = (total * probs[indices]) **(-beta)
        weights /= weights.max()  # 归一化
        weights = torch.tensor(weights, dtype=torch.float32)

        # 转换为批量张量（CPU上）
        batch = Experience(*zip(*experiences))
        states = torch.stack(batch.state)
        actions = torch.tensor(batch.action, dtype=torch.long)
        rewards = torch.tensor(batch.reward, dtype=torch.float32)
        next_states = torch.stack(batch.next_state)
        dones = torch.tensor(batch.done, dtype=torch.float32)

        return (states, actions, rewards, next_states, dones, indices, weights)

    def update_priorities(self, indices, td_errors):
        """用TD误差更新优先级（避免优先级为0）"""
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error.item()) + 1e-6  # 加小值防止概率为0

    def __len__(self):
        return len(self.memory)