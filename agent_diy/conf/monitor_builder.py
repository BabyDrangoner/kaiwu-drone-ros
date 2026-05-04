#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (C) 2026  DIY Author
###########################################################################
"""Monitor panel for DIY agent.

面板名规则: 1~20 字符, 仅中英文/数字/_/-/空格 (不能有括号等其他符号).
指标名规则: 1~40 字符, 支持中英文/数字/_/-/{}/空格.

指标分 4 组:
  1. 结果占比     - 局末结果分布, 直接反映 fail 原因
  2. 平均表现     - 投递 / 步数 / 剩余电量等均值
  3. 痛点诊断     - bounce/stuck/最近 NPC 距离/低电步数
  4. 算法监控     - 损失
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    monitor = MonitorConfigBuilder()

    builder = (
        monitor.title("智运无人机 DIY")

        # ---------------------- 结果占比 ---------------------- #
        .add_group(group_name="结果占比", group_name_en="result_ratio")
        .add_panel(name="总失败率", name_en="fail_rate", type="line")
        .add_metric(metrics_name="fail_rate", expr="avg(fail_rate{})")
        .end_panel()
        .add_panel(name="失败占比 碰NPC", name_en="fail_npc_rate", type="line")
        .add_metric(metrics_name="fail_npc_rate", expr="avg(fail_npc_rate{})")
        .end_panel()
        .add_panel(name="失败占比 耗电", name_en="fail_battery_rate", type="line")
        .add_metric(metrics_name="fail_battery_rate", expr="avg(fail_battery_rate{})")
        .end_panel()
        .add_panel(name="失败占比 其他", name_en="fail_other_rate", type="line")
        .add_metric(metrics_name="fail_other_rate", expr="avg(fail_other_rate{})")
        .end_panel()
        .add_panel(name="成功率 有投递", name_en="win_rate", type="line")
        .add_metric(metrics_name="win_rate", expr="avg(win_rate{})")
        .end_panel()
        .add_panel(name="到终点0投递占比", name_en="win_nodeliver_rate", type="line")
        .add_metric(metrics_name="win_nodeliver_rate", expr="avg(win_nodeliver_rate{})")
        .end_panel()
        .end_group()

        # ---------------------- 平均表现 ---------------------- #
        .add_group(group_name="平均表现", group_name_en="performance")
        .add_panel(name="窗口平均累积回报", name_en="avg_reward", type="line")
        .add_metric(metrics_name="avg_reward", expr="avg(avg_reward{})")
        .end_panel()
        .add_panel(name="窗口平均投递数", name_en="avg_delivered", type="line")
        .add_metric(metrics_name="avg_delivered", expr="avg(avg_delivered{})")
        .end_panel()
        .add_panel(name="窗口平均使用步数", name_en="avg_steps", type="line")
        .add_metric(metrics_name="avg_steps", expr="avg(avg_steps{})")
        .end_panel()
        .add_panel(name="窗口平均剩余电量比", name_en="avg_batt_left_ratio", type="line")
        .add_metric(metrics_name="avg_batt_left_ratio", expr="avg(avg_batt_left_ratio{})")
        .end_panel()
        .add_panel(name="本局累积回报", name_en="reward", type="line")
        .add_metric(metrics_name="reward", expr="avg(reward{})")
        .end_panel()
        .add_panel(name="本局投递数", name_en="delivered", type="line")
        .add_metric(metrics_name="delivered", expr="avg(delivered{})")
        .end_panel()
        .add_panel(name="本局剩余电量", name_en="battery_left", type="line")
        .add_metric(metrics_name="battery_left", expr="avg(battery_left{})")
        .end_panel()
        .end_group()

        # ---------------------- 痛点诊断 ---------------------- #
        .add_group(group_name="痛点诊断", group_name_en="diagnostic")
        .add_panel(name="每局平均撞墙次数", name_en="avg_bounce_per_ep", type="line")
        .add_metric(metrics_name="avg_bounce_per_ep", expr="avg(avg_bounce_per_ep{})")
        .end_panel()
        .add_panel(name="每局平均卡住步数", name_en="avg_stuck_per_ep", type="line")
        .add_metric(metrics_name="avg_stuck_per_ep", expr="avg(avg_stuck_per_ep{})")
        .end_panel()
        .add_panel(name="最小NPC距离均值", name_en="avg_min_npc_dist", type="line")
        .add_metric(metrics_name="avg_min_npc_dist", expr="avg(avg_min_npc_dist{})")
        .end_panel()
        .add_panel(name="低电步数占比", name_en="avg_low_batt_ratio", type="line")
        .add_metric(metrics_name="avg_low_batt_ratio", expr="avg(avg_low_batt_ratio{})")
        .end_panel()
        .end_group()

        # ---------------------- 算法监控 ---------------------- #
        .add_group(group_name="算法指标", group_name_en="algorithm")
        .add_panel(name="总损失", name_en="total_loss", type="line")
        .add_metric(metrics_name="total_loss", expr="avg(total_loss{})")
        .end_panel()
        .add_panel(name="价值损失", name_en="value_loss", type="line")
        .add_metric(metrics_name="value_loss", expr="avg(value_loss{})")
        .end_panel()
        .add_panel(name="策略损失", name_en="policy_loss", type="line")
        .add_metric(metrics_name="policy_loss", expr="avg(policy_loss{})")
        .end_panel()
        .add_panel(name="熵", name_en="entropy_loss", type="line")
        .add_metric(metrics_name="entropy_loss", expr="avg(entropy_loss{})")
        .end_panel()
        .end_group()
    )

    return builder.build()
