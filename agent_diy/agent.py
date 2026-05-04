#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (C) 2026  DIY Author
###########################################################################
"""
DIY Drone Delivery Agent.
"""

import os
import numpy as np
import torch

# CPU 训练: 限制线程, 避免互相争抢
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from kaiwudrl.interface.agent import BaseAgent

from agent_diy.algorithm.algorithm import Algorithm
from agent_diy.conf.conf import Config
from agent_diy.feature.definition import ActData, ObsData
from agent_diy.feature.preprocessor import Preprocessor
from agent_diy.model.model import Model


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(42)
        np.random.seed(42)
        self.device = device if device is not None else torch.device("cpu")

        self.model = Model(self.device).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=Config.START_LR,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.algorithm = Algorithm(
            self.model, self.optimizer, device=self.device, logger=logger, monitor=monitor
        )
        self.preprocessor = Preprocessor()
        self.last_action = -1

        super().__init__(agent_type, device, logger, monitor)

    # ---------------- lifecycle ---------------- #
    def reset(self, env_obs=None):
        self.preprocessor.reset()
        self.last_action = -1

    # ---------------- inference ---------------- #
    def _forward(self, feature):
        self.model.set_eval_mode()
        obs_t = torch.tensor(np.array([feature]), dtype=torch.float32).view(1, Config.DIM_OF_OBSERVATION)
        obs_t = obs_t.to(self.device)
        with torch.no_grad():
            logits, value = self.model(obs_t, inference=True)
        return logits.cpu().numpy()[0], value.cpu().numpy()[0]

    def predict(self, list_obs_data):
        feature = list_obs_data[0].feature
        legal_action = list_obs_data[0].legal_action
        logits, value = self._forward(feature)
        legal_np = np.array(legal_action, dtype=np.float32)
        prob = self._legal_softmax(logits, legal_np)
        action = self._sample(prob, use_max=False)
        d_action = self._sample(prob, use_max=True)
        return [ActData(action=[action], d_action=[d_action], prob=list(prob), value=value)]

    def exploit(self, env_obs):
        obs_data, _ = self.observation_process(env_obs)
        if obs_data is None:
            return 0
        act_data = self.predict([obs_data])[0]
        return self.action_process(act_data, is_stochastic=False)

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    # ---------------- feature ---------------- #
    def observation_process(self, env_obs, extra_info=None):
        feature, legal_action, reward = self.preprocessor.feature_process(env_obs, self.last_action)
        remain_info = {"reward": reward}
        return ObsData(feature=list(feature), legal_action=legal_action), remain_info

    def action_process(self, act_data, is_stochastic=True):
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return self.last_action

    # ---------------- save / load ---------------- #
    def save_model(self, path=None, id="1"):
        if path is None:
            return
        file_path = os.path.join(path, f"model.ckpt-{id}.pkl")
        state = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state, file_path)
        if self.logger:
            self.logger.info(f"save model {file_path}")

    def load_model(self, path=None, id="1"):
        if path is None:
            return
        file_path = os.path.join(path, f"model.ckpt-{id}.pkl")
        if not os.path.exists(file_path):
            return
        state = torch.load(file_path, map_location=self.device)
        # 使用 adapter 加载, 支持老 ckpt (backbone.0 输入 352) 到新网络 (384) 的
        # 部分加载. 新的 action_aux_enc 和 backbone.0 的尾部 32 列保持原状
        # (随机初始化 / 零初始化), 首次 forward 与老模型等价.
        if hasattr(self.model, "adapt_load_state_dict"):
            self.model.adapt_load_state_dict(state, logger=self.logger)
        else:
            self.model.load_state_dict(state, strict=False)
        if self.logger:
            self.logger.info(f"load model {file_path}")

    # ---------------- utils ---------------- #
    @staticmethod
    def _legal_softmax(logits, legal):
        _W = 1e20
        tmp = logits - _W * (1.0 - legal)
        tmp = tmp - tmp.max()
        tmp = np.exp(np.clip(tmp, -50, 50)) * legal
        s = tmp.sum()
        if s < 1e-9:
            # 全掩码意外: 均匀分布
            legal_sum = legal.sum()
            if legal_sum > 0:
                return legal / legal_sum
            return np.ones_like(legal) / len(legal)
        return tmp / s

    @staticmethod
    def _sample(prob, use_max=False):
        if use_max:
            return int(np.argmax(prob))
        # 防御: 修复 nan / 负数
        prob = np.clip(prob, 0.0, 1.0)
        s = prob.sum()
        if s < 1e-9:
            return 0
        prob = prob / s
        return int(np.random.choice(len(prob), p=prob))
