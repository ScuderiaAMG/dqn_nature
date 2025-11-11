import os
import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
from torch.cuda.amp import autocast, GradScaler  # AMP核心组件

from atari_wrappers import make_atari
from model import DQN, huber_loss
from prioritized_replay_buffer import PrioritizedReplayBuffer

# 超参数配置（适配高性能GPU）
BATCH_SIZE = 64  # 批量大小，利用显存优势
GAMMA = 0.99  # 奖励折扣因子
LEARNING_RATE = 1e-4  # 优化器学习率
TAU = 0.001  # 目标网络软更新系数
ALPHA = 0.6  # PER优先级指数
BETA_START = 0.4  # 重要性采样初始权重
BETA_FRAMES = 1000000  # Beta增长至1.0的帧数
BUFFER_SIZE = 1000000  # 经验回放缓冲区容量
EPS_START = 1.0  # ε-greedy初始探索率
EPS_END = 0.1  # 最终探索率
EPS_DECAY = 1000000  # 探索率衰减速率
MAX_FRAMES = 50000000  # 总训练帧数
HARD_UPDATE_FRAMES = 10000  # 目标网络硬更新间隔

def main():
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device} (显存: {torch.cuda.get_device_properties(device).total_memory/1024**3:.1f}GB)")

    # 创建Atari环境
    env = make_atari("BreakoutNoFrameskip-v4")
    obs_shape = env.observation_space.shape  # (4, 84, 84)
    n_actions = env.action_space.n

    # 初始化网络（保持FP32参数，AMP自动处理混合精度）
    policy_net = DQN(obs_shape, n_actions).to(device)
    target_net = DQN(obs_shape, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # 优化器与工具初始化
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = PrioritizedReplayBuffer(BUFFER_SIZE, alpha=ALPHA)
    scaler = GradScaler()  # AMP梯度缩放器

    # 训练状态变量
    frame_idx = 0
    episode_reward = 0
    rewards = []
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.uint8, device=device)  # 单样本形状：(4,84,84)

    while frame_idx < MAX_FRAMES:
        # 1. ε-greedy策略选择动作
        eps = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * frame_idx / EPS_DECAY)
        if np.random.random() < eps:
            action = np.random.randint(n_actions)
        else:
            with torch.no_grad():
                # 推理时启用AMP（自动转为FP16计算）
                with autocast():
                    q_vals = policy_net(obs.unsqueeze(0))  # 添加batch维度：(1,4,84,84)
                action = q_vals.max(1)[1].item()

        # 2. 与环境交互并存储经验
        next_obs, reward, done, truncated, _ = env.step(action)
        reward = np.clip(reward, -1, 1) 
        episode_reward += reward
        done_flag = done or truncated
        next_obs = torch.tensor(next_obs, dtype=torch.uint8, device=device)
        memory.push(obs, action, reward, next_obs, done_flag)
        obs = next_obs

        # 3. 训练步骤（缓冲区数据充足时）
        # 训练步骤（修改后）
        if len(memory) > BATCH_SIZE:
            beta = min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)
            states, actions, rewards_batch, next_states, dones, indices, weights = memory.sample(BATCH_SIZE, beta)
    
        # 输入转换（确保不剥离梯度）
            states = states.to(device, non_blocking=True).float() / 255.0  # 增加归一化
            actions = actions.to(device, non_blocking=True)
            rewards_batch = rewards_batch.to(device, non_blocking=True).float()
            next_states = next_states.to(device, non_blocking=True).float() / 255.0  # 增加归一化
            dones = dones.to(device, non_blocking=True).float()
            weights = weights.to(device, non_blocking=True).float()

            # Double DQN 目标计算
            with torch.no_grad():
                with autocast():  # 在no_grad内使用autocast
                    next_actions = policy_net(next_states).max(1)[1].unsqueeze(1)
                    target_q = target_net(next_states).gather(1, next_actions).squeeze(1)
                target = rewards_batch + (1 - dones) * GAMMA * target_q

            with autocast():    # 当前网络预测（有梯度）
                current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                td_errors = current_q - target  # 保留梯度信息
                loss = (weights * huber_loss(td_errors)).mean()  # 确保损失可微

    #        反向传播（AMP）
            optimizer.zero_grad()
            scaler.scale(loss).backward()  # 此时 loss 应有 grad_fn
            scaler.step(optimizer)
            scaler.update()

    # 更新优先级
            memory.update_priorities(indices, td_errors.detach().cpu().numpy())  # 仅这里 detach
            #loss.backward()
            optimizer.step()

# 新增：删除临时变量并清理CUDA缓存
            del states, actions, rewards, next_states, dones, q_values, next_q_values, loss
            torch.cuda.empty_cache()  # 强制释放未使用的显存
        # 4. 回合结束处理
        if done or truncated:
            rewards.append(episode_reward)
            # 每10回合打印平均奖励
            if len(rewards) % 10 == 0:
                avg_reward = np.mean(rewards[-10:])
                print(f"帧: {frame_idx}, 10回合平均奖励: {avg_reward:.2f}, ε: {eps:.3f}")
            episode_reward = 0
            obs, _ = env.reset()
            obs = torch.tensor(obs, dtype=torch.uint8, device=device)

        # 5. 定期验证AMP状态（每1000帧）
        if frame_idx % 1000 == 0:
            print(f"\nAMP验证 (frame: {frame_idx}):")
            print(f"  网络参数类型: {policy_net.conv1.weight.dtype} (预期float32)")
            # 仅当states存在时验证输入输出类型
            if 'states' in locals():
                print(f"  输入数据类型: {states.dtype} (预期uint8/float32)")
                with autocast():
                    test_out = policy_net(states[:1])
                print(f"  输出数据类型: {test_out.dtype} (预期float16)")
            else:
                print(f"  输入数据类型: 缓冲区数据不足（等待训练启动）")
            print(f"  当前显存占用: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        
        # 在train.py训练循环中添加
        if frame_idx >= 30000000:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 5e-5  # 从1e-4降至5e-5

        # 6. 定期保存模型（每100万帧）
        if frame_idx % 1000000 == 0 and frame_idx > 0:
            os.makedirs("models", exist_ok=True)
            torch.save(policy_net.state_dict(), f"models/dqn_breakout_{frame_idx//1000000}M.pth")
            print(f"模型保存至 models/dqn_breakout_{frame_idx//1000000}M.pth")

        frame_idx += 1

    # 训练结束处理
    env.close()
    os.makedirs("models", exist_ok=True)
    torch.save(policy_net.state_dict(), "models/dqn_breakout_final2.pth")
    print("最终模型保存至 models/dqn_breakout_final2.pth")

if __name__ == "__main__":
    main()