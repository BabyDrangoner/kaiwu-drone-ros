#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (C) 2026  DIY Author
###########################################################################
"""
DIY training workflow (with detailed episode-end diagnostics).
"""

import os
import time
from collections import deque, Counter

import numpy as np

from agent_diy.conf.conf import Config
from agent_diy.feature.definition import SampleData, sample_process
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


SAVE_MODEL_INTERVAL = 600     # 10 分钟保存一次 (微调阶段更频繁落盘)
MONITOR_INTERVAL = 60         # 60s 上报一次
STATS_WINDOW = 100            # 滑动窗口大小: 最近 100 局


# 局末结果枚举
RESULT_FAIL_NPC = "fail_npc"
RESULT_FAIL_BATTERY = "fail_battery"
RESULT_FAIL_OTHER = "fail_other"
RESULT_WIN = "win"
RESULT_WIN_NODELIVER = "win_nodeliver"


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    env, agent = envs[0], agents[0]

    usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        if logger:
            logger.error("usr_conf is None, check agent_diy/conf/train_env_conf.toml")
        return

    runner = EpisodeRunner(env=env, agent=agent, usr_conf=usr_conf, logger=logger, monitor=monitor)
    last_save = time.time()

    while True:
        for g_data in runner.run_episodes():
            if hasattr(agent, "send_sample_data"):
                agent.send_sample_data(g_data)
            else:
                agent.learn(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save >= SAVE_MODEL_INTERVAL:
                agent.save_model()
                last_save = now


# ---------------------------------------------------------------------------- #
# 局末统计器: 维护最近 STATS_WINDOW 局的结果, 计算比率与均值
# ---------------------------------------------------------------------------- #
class EpisodeStatsTracker:
    def __init__(self, window=STATS_WINDOW):
        self.window = window
        self.results = deque(maxlen=window)      # 局末原因列表
        self.rewards = deque(maxlen=window)      # 本局累计 reward
        self.delivered = deque(maxlen=window)
        self.steps = deque(maxlen=window)
        self.batt_left_ratio = deque(maxlen=window)  # 归一化后的剩余电量
        self.bounce_cnts = deque(maxlen=window)
        self.stuck_cnts = deque(maxlen=window)
        self.min_npc_dists = deque(maxlen=window)
        self.low_batt_ratios = deque(maxlen=window)  # 本局低电步数 / 总步数

    def add(self, *, result, reward, delivered, steps, batt_left, batt_max,
            bounce, stuck, min_npc_dist, low_batt_steps):
        self.results.append(result)
        self.rewards.append(float(reward))
        self.delivered.append(int(delivered))
        self.steps.append(int(steps))
        self.batt_left_ratio.append(float(batt_left) / max(batt_max, 1))
        self.bounce_cnts.append(int(bounce))
        self.stuck_cnts.append(int(stuck))
        self.min_npc_dists.append(float(min_npc_dist) if min_npc_dist is not None else -1.0)
        self.low_batt_ratios.append(float(low_batt_steps) / max(steps, 1))

    def as_monitor_dict(self):
        n = max(len(self.results), 1)
        cnt = Counter(self.results)

        def _safe_mean(seq, filter_neg=False):
            arr = [v for v in seq if (not filter_neg) or v >= 0]
            return round(float(np.mean(arr)), 4) if arr else 0.0

        return {
            # 结果占比 (百分比, 0~100, 看起来直观)
            "fail_rate": round(100.0 * (cnt[RESULT_FAIL_NPC] + cnt[RESULT_FAIL_BATTERY] + cnt[RESULT_FAIL_OTHER]) / n, 2),
            "fail_npc_rate": round(100.0 * cnt[RESULT_FAIL_NPC] / n, 2),
            "fail_battery_rate": round(100.0 * cnt[RESULT_FAIL_BATTERY] / n, 2),
            "fail_other_rate": round(100.0 * cnt[RESULT_FAIL_OTHER] / n, 2),
            "win_rate": round(100.0 * cnt[RESULT_WIN] / n, 2),
            "win_nodeliver_rate": round(100.0 * cnt[RESULT_WIN_NODELIVER] / n, 2),
            # 平均表现
            "avg_reward": round(float(np.mean(self.rewards)), 4) if self.rewards else 0.0,
            "avg_delivered": round(float(np.mean(self.delivered)), 4) if self.delivered else 0.0,
            "avg_steps": round(float(np.mean(self.steps)), 2) if self.steps else 0.0,
            "avg_batt_left_ratio": round(float(np.mean(self.batt_left_ratio)), 4) if self.batt_left_ratio else 0.0,
            # 诊断(痛点)
            "avg_bounce_per_ep": round(float(np.mean(self.bounce_cnts)), 2) if self.bounce_cnts else 0.0,
            "avg_stuck_per_ep": round(float(np.mean(self.stuck_cnts)), 2) if self.stuck_cnts else 0.0,
            "avg_min_npc_dist": _safe_mean(self.min_npc_dists, filter_neg=True),
            "avg_low_batt_ratio": round(float(np.mean(self.low_batt_ratios)), 4) if self.low_batt_ratios else 0.0,
            "window_size": len(self.results),
        }


def classify_result(terminated, truncated, delivered, batt_left, last_npc_dist, result_code):
    """根据最后一帧状态推断局末原因.

    优先级:
      1. truncated: WIN / WIN_NODELIVER
      2. terminated & batt_left <= 0: FAIL_BATTERY
      3. terminated & last_npc_dist <= 2: FAIL_NPC
      4. terminated 其他: FAIL_OTHER (极少见, 兜底)
    """
    if truncated and not terminated:
        return RESULT_WIN if delivered > 0 else RESULT_WIN_NODELIVER
    if terminated:
        if batt_left is not None and batt_left <= 0:
            return RESULT_FAIL_BATTERY
        if last_npc_dist is not None and last_npc_dist <= 2.0:
            return RESULT_FAIL_NPC
        return RESULT_FAIL_OTHER
    # 既没 terminated 也没 truncated, 理论上不该到这里
    return RESULT_FAIL_OTHER


# ---------------------------------------------------------------------------- #
# 单局运行器
# ---------------------------------------------------------------------------- #
class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor

        self.episode_cnt = 0
        self.last_monitor_time = 0
        self.last_metrics_time = 0
        self.stats = EpisodeStatsTracker(window=STATS_WINDOW)

    def run_episodes(self):
        while True:
            # 周期性打印框架 metrics
            now = time.time()
            if now - self.last_metrics_time >= MONITOR_INTERVAL:
                m = get_training_metrics()
                self.last_metrics_time = now
                if m and self.logger:
                    self.logger.info(f"training_metrics: {m}")

            env_obs = self.env.reset(self.usr_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            obs_data, _ = self.agent.observation_process(env_obs)
            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0
            total_delivered = 0

            if self.logger:
                self.logger.info(f"episode {self.episode_cnt} start, conf={self.usr_conf}")

            while not done:
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                act = self.agent.action_process(act_data)

                env_reward, env_obs = self.env.step(act)
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                step += 1
                done = terminated or truncated

                _obs_data, _remain_info = self.agent.observation_process(env_obs)
                reward_list = _remain_info.get("reward", [0.0]) if _remain_info else [0.0]
                reward = np.array(reward_list, dtype=np.float32)
                if reward.shape[0] < Config.VALUE_NUM:
                    reward = np.pad(reward, (0, Config.VALUE_NUM - reward.shape[0]))

                # -------- 局末处理 -------- #
                final_reward = np.zeros(Config.VALUE_NUM, dtype=np.float32)
                if done:
                    pp = self.agent.preprocessor
                    total_delivered = pp.delivered
                    # env_obs["observation"] / ["extra_info"] 一般是 dict,
                    # 但为了防御加类型检查, 避免再次踩到 'list' has no .get
                    obs_blk = env_obs.get("observation", {}) if isinstance(env_obs, dict) else {}
                    info_block = obs_blk.get("env_info", {}) if isinstance(obs_blk, dict) else {}
                    extra_info = env_obs.get("extra_info", {}) if isinstance(env_obs, dict) else {}
                    if not isinstance(extra_info, dict):
                        extra_info = {}
                    total_score = info_block.get("total_score", 0) if isinstance(info_block, dict) else 0
                    result_code = extra_info.get("result_code", 0)
                    result_msg = extra_info.get("result_message", "")

                    # 推断结果
                    result_type = classify_result(
                        terminated=terminated,
                        truncated=truncated,
                        delivered=total_delivered,
                        batt_left=pp.battery,
                        last_npc_dist=pp.last_npc_dist,
                        result_code=result_code,
                    )

                    if result_type in (RESULT_FAIL_NPC, RESULT_FAIL_BATTERY, RESULT_FAIL_OTHER):
                        final_reward[0] = Config.RewardWeights.TERMINATED_FAIL
                    elif result_type == RESULT_WIN_NODELIVER:
                        final_reward[0] = Config.RewardWeights.TRUNCATED_NODELIVER
                    else:  # RESULT_WIN
                        final_reward[0] = 0.0

                    # 更新滑动窗口
                    self.stats.add(
                        result=result_type,
                        reward=total_reward + float(final_reward[0]),
                        delivered=total_delivered,
                        steps=step,
                        batt_left=pp.battery,
                        batt_max=pp.battery_max,
                        bounce=pp.ep_bounce_count,
                        stuck=pp.ep_stuck_count,
                        min_npc_dist=pp.ep_min_npc_dist,
                        low_batt_steps=pp.ep_low_batt_steps,
                    )

                    if self.logger:
                        self.logger.info(
                            f"[DONE] ep:{self.episode_cnt} step:{step} "
                            f"result:{result_type} delivered:{total_delivered} "
                            f"score:{total_score} reward:{total_reward:.2f}+{final_reward[0]:.2f} "
                            f"batt:{pp.battery}/{pp.battery_max} "
                            f"bounce:{pp.ep_bounce_count} stuck:{pp.ep_stuck_count} "
                            f"min_npc:{pp.ep_min_npc_dist} "
                            f"result_code:{result_code} msg:{result_msg}"
                        )

                total_reward += float(reward.sum())

                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array([act_data.action[0]], dtype=np.float32),
                    reward=reward,
                    done=np.array([float(done)], dtype=np.float32),
                    value=np.array(act_data.value, dtype=np.float32).flatten()[: Config.VALUE_NUM],
                    next_value=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                    advantage=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                    prob=np.array(act_data.prob, dtype=np.float32),
                    reward_sum=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                )
                collector.append(frame)

                if done:
                    if collector:
                        collector[-1].reward = collector[-1].reward + final_reward

                    # -------- 监控上报 (60s 节流) -------- #
                    if self.monitor and (time.time() - self.last_monitor_time >= MONITOR_INTERVAL):
                        report = self.stats.as_monitor_dict()
                        # 本局瞬时值, 辅助观察
                        report.update(
                            {
                                "reward": round(total_reward + float(final_reward[0]), 4),
                                "episode_cnt": self.episode_cnt,
                                "delivered": total_delivered,
                                "steps_used": step,
                                "battery_left": self.agent.preprocessor.battery,
                            }
                        )
                        self.monitor.put_data({os.getpid(): report})
                        if self.logger:
                            self.logger.info(f"[STATS] {report}")
                        self.last_monitor_time = time.time()

                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                obs_data = _obs_data
