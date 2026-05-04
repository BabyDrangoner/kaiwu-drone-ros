#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (C) 2026  DIY Author
###########################################################################
"""
DIY feature/sample definitions.
"""

import numpy as np

from common_python.utils.common_func import create_cls
from agent_diy.conf.conf import Config


# Observation -> model input
ObsData = create_cls("ObsData", feature=None, legal_action=None)

# Model output -> action
ActData = create_cls(
    "ActData",
    action=None,       # 采样动作
    d_action=None,     # 贪心动作
    prob=None,         # 动作概率分布(用于 PPO old_prob)
    value=None,        # 价值
)

# Training sample
SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,
    legal_action=Config.ACTION_NUM,
    act=1,
    reward=Config.VALUE_NUM,
    done=1,
    value=Config.VALUE_NUM,
    next_value=Config.VALUE_NUM,
    advantage=Config.VALUE_NUM,
    prob=Config.ACTION_NUM,
    reward_sum=Config.VALUE_NUM,
)


def sample_process(list_sample_data):
    """Fill next_value and compute GAE advantage + reward_sum."""
    n = len(list_sample_data)
    if n == 0:
        return list_sample_data

    # next_value
    for i in range(n - 1):
        list_sample_data[i].next_value = list_sample_data[i + 1].value
    # 最后一步 next_value 保持初始化的 0(done=True 时也对)

    _calc_gae(list_sample_data)
    return list_sample_data


def _calc_gae(samples):
    gae = 0.0
    gamma = Config.GAMMA
    lam = Config.LAMDA
    for s in reversed(samples):
        done = float(s.done[0]) if hasattr(s.done, "__len__") else float(s.done)
        reward = s.reward[0] if hasattr(s.reward, "__len__") else s.reward
        value = s.value[0] if hasattr(s.value, "__len__") else s.value
        next_value = s.next_value[0] if hasattr(s.next_value, "__len__") else s.next_value

        delta = reward + gamma * next_value * (1.0 - done) - value
        gae = delta + gamma * lam * (1.0 - done) * gae
        s.advantage = np.array([gae], dtype=np.float32)
        s.reward_sum = np.array([gae + value], dtype=np.float32)


def reward_shaping(frame_no, score, terminated, truncated, remain_info, _remain_info, obs, _obs):
    """Reward shaping entry (not strictly needed since preprocessor computes reward)."""
    if _remain_info and "reward" in _remain_info:
        return _remain_info["reward"]
    return 0.0
