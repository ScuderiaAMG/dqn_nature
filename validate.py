import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt

# 从工程中导入环境包装器（保持与训练一致的预处理）
from atari_wrappers import make_atari


# -------------------------- 重新定义与权重匹配的DQN类 --------------------------
# 该类使用nn.Sequential，层命名为conv.0、conv.2、fc.0等，与保存的权重文件匹配
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.input_shape = input_shape  # 输入形状：(4, 84, 84)
        self.num_actions = num_actions

        # 卷积层（nn.Sequential自动命名为conv.0, conv.2, conv.4...）
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),  # conv.0
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # conv.2
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # conv.4
            nn.ReLU()
        )

        # 计算卷积层输出维度
        self.conv_out_size = self._get_conv_out(input_shape)

        # 全连接层（nn.Sequential自动命名为fc.0, fc.2...）
        self.fc = nn.Sequential(
            nn.Linear(self.conv_out_size, 512),  # fc.0
            nn.ReLU(),
            nn.Linear(512, num_actions)  # fc.2
        )

    def _get_conv_out(self, shape):
        """计算卷积层输出的扁平化维度"""
        with torch.no_grad():
            return self.conv(torch.zeros(1, *shape)).view(1, -1).size(1)

    def forward(self, x):
        x = x.float() / 255.0  # 归一化到[0,1]（与训练时预处理一致）
        conv_out = self.conv(x).view(x.size(0), -1)  # 扁平化
        return self.fc(conv_out)


# -------------------------- 论文基准数据 --------------------------
PAPER_BASELINES = {
    "BreakoutNoFrameskip-v4": {
        "random": 0.1,        # 随机策略得分
        "human": 4.3,         # 人类水平得分
        "dqn_paper": 31.7,    # 论文中DQN得分
        "std_dev": 8.4        # 论文结果标准差
    },
    "PongNoFrameskip-v4": {
        "random": -20.7,
        "human": 9.3,
        "dqn_paper": 18.9,
        "std_dev": 0.3
    }
}


# -------------------------- 辅助函数 --------------------------
def calculate_performance_metrics(agent_scores, paper_data):
    """计算性能指标并与论文对比"""
    metrics = {
        "mean_score": np.mean(agent_scores),
        "std_score": np.std(agent_scores),
        "min_score": np.min(agent_scores),
        "max_score": np.max(agent_scores),
        "median_score": np.median(agent_scores)
    }
    
    if paper_data:
        metrics["human_ratio"] = (metrics["mean_score"] / paper_data["human"]) * 100
        metrics["paper_ratio"] = (metrics["mean_score"] / paper_data["dqn_paper"]) * 100
    
    return metrics


def save_results(metrics, env_id, video_dir, paper_data=None):
    """保存结果与可视化对比图"""
    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": env_id,
        "metrics": metrics,
        "paper_comparison": paper_data
    }
    
    # 保存JSON结果
    result_path = os.path.join(video_dir, "validation_results.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=4)
    
    # 生成对比图表
    if paper_data:
        plt.figure(figsize=(10, 6))
        plt.bar(
            ["随机策略", "本模型", "论文DQN", "人类水平"],
            [
                paper_data["random"],
                metrics["mean_score"],
                paper_data["dqn_paper"],
                paper_data["human"]
            ]
        )
        plt.title(f"{env_id} 性能对比")
        plt.ylabel("平均得分")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.savefig(os.path.join(video_dir, "performance_comparison.png"))
        plt.close()
    
    return result


# -------------------------- 主验证函数 --------------------------
def validate_model(
    model_path,
    env_id="BreakoutNoFrameskip-v4",
    num_episodes=30,
    max_steps_per_episode=18000,  # 5分钟(60fps)
    epsilon=0.05  # 评估探索率（论文指定）
):
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建视频目录（带时间戳避免覆盖）
    video_dir = f"validation_videos/{env_id}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(video_dir, exist_ok=True)

    # 创建环境（与训练时预处理一致）
    env = make_atari(env_id)
    env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda x: True,  # 录制所有回合
        name_prefix="dqn_validation"
    )

    # 环境参数
    obs_shape = env.observation_space.shape  # (4,84,84)
    n_actions = env.action_space.n
    print(f"环境: {env_id} | 观测形状: {obs_shape} | 动作数: {n_actions}")

    # 初始化模型（使用当前脚本中定义的兼容版DQN）
    model = DQN(obs_shape, n_actions).to(device)
    
    # 加载模型权重
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()  # 评估模式
        print(f"✅ 成功加载模型: {model_path}")
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        env.close()
        return

    # 论文基准数据
    paper_data = PAPER_BASELINES.get(env_id, None)
    print(f"论文对比数据: {'已加载' if paper_data else '未找到'}\n")

    # 执行验证
    scores = []
    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done and steps < max_steps_per_episode:
            # 转换观测为模型输入格式
            obs_tensor = torch.tensor(obs, dtype=torch.uint8, device=device).unsqueeze(0)

            # 选择动作（ε-贪婪）
            if np.random.random() < epsilon:
                action = env.action_space.sample()  # 随机探索
            else:
                with torch.no_grad():  # 禁用梯度计算
                    q_values = model(obs_tensor)
                    action = q_values.max(1)[1].item()  # 贪心选择

            # 执行动作
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            obs = next_obs
            steps += 1
            done = terminated or truncated

        scores.append(total_reward)
        print(f"回合 {episode:2d}/{num_episodes} | 得分: {total_reward:6.1f} | 步数: {steps}")

    # 清理环境
    env.close()

    # 结果处理
    metrics = calculate_performance_metrics(scores, paper_data)
    save_results(metrics, env_id, video_dir, paper_data)

    # 打印总结
    print("\n" + "="*60)
    print(f"验证总结: {env_id}")
    print(f"平均得分: {metrics['mean_score']:.2f} ± {metrics['std_score']:.2f}")
    print(f"最高/最低得分: {metrics['max_score']:.2f} / {metrics['min_score']:.2f}")
    
    if paper_data:
        print(f"\n与论文对比:")
        print(f"人类水平: {paper_data['human']} (100%)")
        print(f"本模型性能: {metrics['human_ratio']:.1f}% 人类水平")
        print(f"论文DQN性能: {paper_data['dqn_paper'] / paper_data['human'] * 100:.1f}% 人类水平")
    
    print(f"\n视频保存路径: {os.path.abspath(video_dir)}")
    print("="*60)

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DQN模型验证（与论文对比）")
    parser.add_argument("--model_path", type=str, required=True, help="模型权重路径")
    parser.add_argument("--env_id", type=str, default="BreakoutNoFrameskip-v4", help="环境ID")
    parser.add_argument("--num_episodes", type=int, default=30, help="验证回合数")
    args = parser.parse_args()

    validate_model(
        model_path=args.model_path,
        env_id=args.env_id,
        num_episodes=args.num_episodes
    )