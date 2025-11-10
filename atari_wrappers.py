# #import gymnasium as gym
# #import numpy as np
# #import cv2

# #class NoopResetEnv(gym.Wrapper):
# #    def __init__(self, env, noop_max=30):
# #        super().__init__(env)
# #        self.noop_max = noop_max
# #        self.noop_action = 0

# #    def reset(self, **kwargs):
# #        obs, info = self.env.reset(**kwargs)
# #        noops = np.random.randint(1, self.noop_max + 1)
# #        for _ in range(noops):
# #            obs, _, done, _, _ = self.env.step(self.noop_action)
# #            if done:
# #                obs, info = self.env.reset(**kwargs)
# #        return obs, info

# #class MaxAndSkipEnv(gym.Wrapper):
# #    def __init__(self, env, skip=4):
# #        super().__init__(env)
# #        self._skip = skip
# #        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)

# #    def step(self, action):
# #        total_reward = 0.0
# #        done = None
# #        for i in range(self._skip):
# #            obs, reward, done, truncated, info = self.env.step(action)
# #            if i == self._skip - 2: self._obs_buffer[0] = obs
# #            if i == self._skip - 1: self._obs_buffer[1] = obs
# #            total_reward += reward
# #            if done or truncated:
# #                break
# #        max_frame = self._obs_buffer.max(axis=0)
# #        return max_frame, total_reward, done, truncated, info

# #class WarpFrame(gym.ObservationWrapper):
# #    def __init__(self, env):
# #        super().__init__(env)
# #        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

# #    def observation(self, frame):
# #        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
# #        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
# #        return frame[:, :, None]
# #class WarpFrame(gym.ObservationWrapper):
# #    def __init__(self, env, width=84, height=84):
# #        super().__init__(env)
# #        self.width = width
# #        self.height = height
#         # 输出 (84, 84)，不是 (84, 84, 1)
# #        self.observation_space = gym.spaces.Box(
# #            low=0, high=255, shape=(height, width), dtype=np.uint8
# #        )

# #    def observation(self, frame):
# #        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
# #        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
# #        return frame  # shape: (84, 84)

# #class FrameStack(gym.Wrapper):
# #    def __init__(self, env, k=4):
# #        super().__init__(env)
# #        self.k = k
# #        self.frames = np.zeros((k,) + env.observation_space.shape, dtype=np.uint8)
# #        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(k,) + env.observation_space.shape[1:], dtype=np.uint8)

# #    def reset(self, **kwargs):
# #        obs, info = self.env.reset(**kwargs)
# #        self.frames = np.tile(obs, (self.k, 1, 1, 1))
# #        return self.frames, info

# #    def step(self, action):
# #        obs, reward, done, truncated, info = self.env.step(action)
# #        self.frames = np.roll(self.frames, shift=-1, axis=0)
# #        self.frames[-1] = obs
# #        return self.frames, reward, done, truncated, info
# #class FrameStack(gym.Wrapper):
# #    def __init__(self, env, k=4):
# #        super().__init__(env)
# #        self.k = k
#         # 假设 env.observation_space.shape 是 (84, 84)
# #        self.frames = np.zeros((k,) + env.observation_space.shape, dtype=np.uint8)
# #        self.observation_space = gym.spaces.Box(
# #            low=0, high=255, shape=(k,) + env.observation_space.shape, dtype=np.uint8
# #        )

# #    def reset(self, **kwargs):
# #        obs, info = self.env.reset(**kwargs)
# #        self.frames = np.tile(obs, (self.k, 1, 1))  # (4, 84, 84)
# #        return self.frames, info

# #    def step(self, action):
# #        obs, reward, done, truncated, info = self.env.step(action)
# #        self.frames = np.roll(self.frames, shift=-1, axis=0)
# #        self.frames[-1] = obs
# #        return self.frames, reward, done, truncated, info

# #def make_atari(env_id):
# #    env = gym.make(env_id, render_mode="rgb_array")
# #    env = NoopResetEnv(env)
# #    env = MaxAndSkipEnv(env)
# #    env = WarpFrame(env)
# #    env = FrameStack(env)
# #    return env
# # 保持原有的NoopResetEnv、MaxAndSkipEnv等包装器
# # 确保FrameStack输出正确的张量形状（与DQN输入匹配）
# import gymnasium as gym
# import numpy as np
# from gymnasium.spaces import Box
# from collections import deque
# import torch
# from PIL import Image

# class NoopResetEnv(gym.Wrapper):
#     def __init__(self, env, noop_max=30):
#         super().__init__(env)
#         self.noop_max = noop_max
#         self.override_num_noops = None
#         self.noop_action = 0
#         assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

#     def reset(self, **kwargs):
#         self.env.reset(** kwargs)
#         if self.override_num_noops is not None:
#             noops = self.override_num_noops
#         else:
#             noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
#         assert noops > 0
#         obs = None
#         for _ in range(noops):
#             obs, _, done, _, _ = self.env.step(self.noop_action)
#             if done:
#                 obs, _ = self.env.reset(**kwargs)
#         return obs, {}

# class MaxAndSkipEnv(gym.Wrapper):
#     def __init__(self, env, skip=4):
#         super().__init__(env)
#         self._skip = skip

#     def step(self, action):
#         total_reward = 0.0
#         done = False
#         for _ in range(self._skip):
#             obs, reward, done, truncated, info = self.env.step(action)
#             total_reward += reward
#             if done or truncated:
#                 break
#         return obs, total_reward, done, truncated, info

# class WarpFrame(gym.ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.size = 84
#         self.observation_space = Box(low=0, high=255, shape=(self.size, self.size), dtype=np.uint8)

#     def observation(self, frame):
#         frame = np.mean(frame, axis=2).astype(np.uint8)  # 转灰度图
#         frame = np.array(Image.fromarray(frame).resize((self.size, self.size), Image.BILINEAR), dtype=np.uint8)
#         return frame

# class FrameStack(gym.ObservationWrapper):
#     def __init__(self, env, k=4):
#         super().__init__(env)
#         self.k = k
#         self.frames = deque([], maxlen=k)
#         shp = env.observation_space.shape
#         self.observation_space = Box(low=0, high=255, shape=(k, shp[0], shp[1]), dtype=np.uint8)

#     def reset(self,** kwargs):
#         obs, info = self.env.reset(**kwargs)
#         for _ in range(self.k):
#             self.frames.append(obs)
#         return self._get_observation(), info

#     def observation(self, obs):
#         self.frames.append(obs)
#         return self._get_observation()

#     def _get_observation(self):
#         return np.stack(list(self.frames), axis=0)  # 输出形状为(4,84,84)

# def make_atari(env_id):
#     env = gym.make(env_id, render_mode="rgb_array")  # 支持视频录制
#     env = NoopResetEnv(env)
#     env = MaxAndSkipEnv(env, skip=4)
#     env = WarpFrame(env)
#     env = FrameStack(env, k=4)
#     return env
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from collections import deque
from PIL import Image  # 修复Image未导入问题

class NoopResetEnv(gym.Wrapper):
    """初始随机执行0-30步NOOP动作，增加随机性"""
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self,** kwargs):
        self.env.reset(**kwargs)
        noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        obs = None
        for _ in range(noops):
            obs, _, done, _, _ = self.env.step(self.noop_action)
            if done:
                obs, _ = self.env.reset(** kwargs)
        return obs, {}

class MaxAndSkipEnv(gym.Wrapper):
    """每4步取最大像素值，减少动作频率"""
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done or truncated:
                break
        return obs, total_reward, done, truncated, info

class WarpFrame(gym.ObservationWrapper):
    """将图像转为84×84灰度图"""
    def __init__(self, env):
        super().__init__(env)
        self.size = 84
        self.observation_space = Box(low=0, high=255, shape=(self.size, self.size), dtype=np.uint8)

    def observation(self, frame):
        frame = np.mean(frame, axis=2).astype(np.uint8)  # 转灰度
        frame = np.array(Image.fromarray(frame).resize((self.size, self.size), Image.BILINEAR), dtype=np.uint8)
        return frame

class FrameStack(gym.ObservationWrapper):
    """堆叠4帧作为状态，捕捉时间信息"""
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = Box(low=0, high=255, shape=(k, shp[0], shp[1]), dtype=np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(** kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_observation(), info

    def observation(self, obs):
        self.frames.append(obs)
        return self._get_observation()

    def _get_observation(self):
        return np.stack(list(self.frames), axis=0)  # 输出形状：(4,84,84)

def make_atari(env_id):
    """创建预处理后的Atari环境（支持视频录制）"""
    env = gym.make(env_id, render_mode="rgb_array")
    env = NoopResetEnv(env)
    env = MaxAndSkipEnv(env, skip=4)
    env = WarpFrame(env)
    env = FrameStack(env, k=4)
    return env