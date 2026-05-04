#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (C) 2026  DIY Author
###########################################################################
"""
DIY PPO algorithm.

- Policy loss: clipped surrogate
- Value loss:  clipped value
- Entropy regularization (masked softmax)
- Advantage normalization per batch
"""

import os
import time

import torch
import torch.nn.functional as F

from agent_diy.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, scheduler=None, device=None, logger=None, monitor=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.monitor = monitor

        self.label_size = Config.ACTION_NUM
        self.value_num = Config.VALUE_NUM
        self.clip_param = Config.CLIP_PARAM
        self.vf_coef = Config.VF_COEF
        self.beta = Config.BETA_START

        self.last_report_time = 0

    # ------------------------------------------------------------------ #
    def learn(self, list_sample_data):
        if not list_sample_data:
            return {"total_loss": 0.0}

        obs = torch.stack([f.obs for f in list_sample_data]).to(self.device)
        legal = torch.stack([f.legal_action for f in list_sample_data]).to(self.device)
        act = torch.stack([f.act for f in list_sample_data]).to(self.device).view(-1, 1).long()
        old_prob = torch.stack([f.prob for f in list_sample_data]).to(self.device)
        adv = torch.stack([f.advantage for f in list_sample_data]).to(self.device).view(-1)
        old_value = torch.stack([f.value for f in list_sample_data]).to(self.device).view(-1)
        reward_sum = torch.stack([f.reward_sum for f in list_sample_data]).to(self.device).view(-1)
        reward = torch.stack([f.reward for f in list_sample_data]).to(self.device).view(-1)

        # Advantage normalization
        if adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        logits, value_pred = self.model(obs)
        total_loss, info = self._compute_loss(
            logits, value_pred.view(-1), legal, act, old_prob, adv, old_value, reward_sum
        )
        total_loss.backward()
        if Config.USE_GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP_RANGE)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # periodic monitor
        now = time.time()
        if now - self.last_report_time >= 60:
            results = {
                "total_loss": round(total_loss.item(), 4),
                "value_loss": round(info["value_loss"], 4),
                "policy_loss": round(info["policy_loss"], 4),
                "entropy_loss": round(info["entropy_loss"], 4),
                "reward": round(reward.mean().item(), 4),
            }
            if self.logger:
                self.logger.info(
                    "[LEARN] policy={policy_loss} value={value_loss} "
                    "ent={entropy_loss} reward={reward}".format(**results)
                )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_time = now

        return {"total_loss": total_loss.item()}

    # ------------------------------------------------------------------ #
    def _compute_loss(self, logits, value_pred, legal, act, old_prob, adv, old_value, reward_sum):
        # 1. Masked softmax
        prob = self._masked_softmax(logits, legal)
        log_prob = torch.log(prob.clamp(1e-9))

        # 2. Policy loss (PPO clip)
        act_idx = act[:, 0]
        new_act_prob = prob.gather(1, act.view(-1, 1)).view(-1).clamp(1e-9)
        old_act_prob = old_prob.gather(1, act.view(-1, 1)).view(-1).clamp(1e-9)
        ratio = new_act_prob / old_act_prob

        surr1 = ratio * adv
        surr2 = ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        # 3. Value loss (clipped)
        v_clip = old_value + (value_pred - old_value).clamp(-self.clip_param, self.clip_param)
        vloss1 = (value_pred - reward_sum) ** 2
        vloss2 = (v_clip - reward_sum) ** 2
        value_loss = 0.5 * torch.max(vloss1, vloss2).mean()

        # 4. Entropy
        entropy = -(prob * log_prob).sum(dim=1).mean()

        total_loss = policy_loss + self.vf_coef * value_loss - self.beta * entropy

        info = {
            "value_loss": value_loss.detach().item(),
            "policy_loss": policy_loss.detach().item(),
            "entropy_loss": entropy.detach().item(),
        }
        return total_loss, info

    @staticmethod
    def _masked_softmax(logits, legal):
        legal = legal.to(logits.dtype)
        logits = logits - logits.max(dim=1, keepdim=True).values
        logits = logits + (legal - 1.0) * 1e5
        return F.softmax(logits, dim=1)
