#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (C) 2026  DIY Author
###########################################################################
"""
DIY Actor-Critic model (v1 + action_aux adapter).

结构要点:
- 特征按类型分段(self / station / npc / charger / warehouse / map / action_aux)
- 实体类型用共享 MLP 编码 + masked mean+max pool
- 局部地图 + 动作辅助信号 各自独立 MLP 编码
- 所有 embedding concat -> backbone(2 层 MLP) -> actor / critic 双头

续训兼容 (关键):
- 老 ckpt (788792) 的 backbone[0] 输入维度是 352.
  新网络输入是 352 + 32 (action_aux_embed) = 384.
- `adapt_load_state_dict` 负责把老 state_dict 的前 352 列拷到新 backbone
  的前 352 列, 最后 32 列保持 0. 这样 load 后首次 forward 的输出和
  老模型在老特征上的输出**数值上完全一致**, 零扰动续训.

参数量:
  旧版 ~77K
  新增 action_aux_enc: 27->32->32 ~2K
  新增 backbone[0] 扩容: 32 * 128 = 4K
  总计 ~83K (可以忽略的 CPU forward 开销)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent_diy.conf.conf import Config


def _fc(in_dim, out_dim, gain=1.0):
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class EntityEncoder(nn.Module):
    """Shared MLP for each slot, then masked mean+max-pool over K slots.

    Input : (B, K, F)
    Output: (B, 2*out_dim)
    """

    def __init__(self, feat_dim, out_dim):
        super().__init__()
        self.fc1 = _fc(feat_dim, out_dim)
        self.fc2 = _fc(out_dim, out_dim)

    def forward(self, x):
        mask = x[..., 0:1]                     # (B, K, 1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = h * mask                           # 空 slot 全 0
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = h.sum(dim=1) / denom

        neg_inf_mask = (1.0 - mask) * -1e4
        h_max = (h + neg_inf_mask).max(dim=1).values
        any_found = (mask.sum(dim=1) > 0).float()
        h_max = h_max * any_found
        return torch.cat([pooled, h_max], dim=-1)   # (B, 2*out_dim)


# ---------------- 旧 backbone concat_dim (用来兼容老 ckpt) ---------------- #
# self(64) + station(2*32) + npc(2*32) + charger(2*32) + warehouse(2*32) + map(32)
# = 64 + 64*4 + 32 = 352
LEGACY_BACKBONE_IN_DIM = 352


class Model(nn.Module):
    """Drone Delivery DIY Actor-Critic (v1 + action_aux adapter)."""

    def __init__(self, device=None):
        super().__init__()
        self.device = device
        self.model_name = "drone_delivery_diy_v1aux"

        C = Config

        # --- 实体编码器 ---
        self.station_enc = EntityEncoder(C.STATION_FEAT, C.ENTITY_EMBED_DIM)      # out: 64
        self.npc_enc = EntityEncoder(C.NPC_FEAT, C.ENTITY_EMBED_DIM)              # out: 64
        self.charger_enc = EntityEncoder(C.CHARGER_FEAT, C.ENTITY_EMBED_DIM)      # out: 64
        self.warehouse_enc = EntityEncoder(C.WAREHOUSE_FEAT, C.ENTITY_EMBED_DIM)  # out: 64

        # --- 自身状态 / 历史 / meta ---
        self.self_enc = nn.Sequential(
            _fc(C.SELF_TOTAL, 64),
            nn.ReLU(),
            _fc(64, 64),
            nn.ReLU(),
        )

        # --- 局部地图 ---
        self.map_enc = nn.Sequential(
            _fc(C.LOCAL_MAP_DIM, C.MAP_EMBED_DIM),
            nn.ReLU(),
        )

        # --- 【新增】动作辅助信号 encoder (27 -> 32) ---
        # 这是一个 2 层 MLP, 因为动作辅助信号维度高 (27) 且语义复杂
        # (risk, block, velocity, polar TTC 四种信号混在一起), 需要非线性组合.
        self.action_aux_enc = nn.Sequential(
            _fc(C.ACTION_AUX_DIM, C.ACTION_AUX_EMBED_DIM),
            nn.ReLU(),
            _fc(C.ACTION_AUX_EMBED_DIM, C.ACTION_AUX_EMBED_DIM),
            nn.ReLU(),
        )

        # --- 主干 ---
        # 新拼接维度: legacy(352) + action_aux(32) = 384
        concat_dim = LEGACY_BACKBONE_IN_DIM + C.ACTION_AUX_EMBED_DIM
        assert concat_dim == 384, f"concat_dim = {concat_dim}, expect 384"

        self.backbone = nn.Sequential(
            _fc(concat_dim, 128),
            nn.ReLU(),
            _fc(128, 128),
            nn.ReLU(),
        )
        # 【关键】backbone 第一层最后 32 列 (对应 action_aux) 初始化为 0,
        # 这样加载老 pkl 后, 老特征那 352 列的计算结果与老模型完全一致,
        # 而 action_aux 的贡献在训练开始时为 0, 由 PPO 在续训中慢慢学大.
        with torch.no_grad():
            self.backbone[0].weight[:, LEGACY_BACKBONE_IN_DIM:].zero_()

        # --- 双头 ---
        self.actor_head = _fc(128, C.ACTION_NUM, gain=0.01)
        self.critic_head = _fc(128, C.VALUE_NUM, gain=1.0)

    # ---------------- helpers ---------------- #
    def _split(self, x):
        """按 offset 切片并 reshape 实体段."""
        C = Config
        self_seg = x[:, : C.SELF_TOTAL]
        station_seg = x[:, C.OFFSET_STATION : C.OFFSET_NPC].view(-1, C.K_STATION, C.STATION_FEAT)
        npc_seg = x[:, C.OFFSET_NPC : C.OFFSET_CHARGER].view(-1, C.K_NPC, C.NPC_FEAT)
        charger_seg = x[:, C.OFFSET_CHARGER : C.OFFSET_WAREHOUSE].view(-1, C.K_CHARGER, C.CHARGER_FEAT)
        warehouse_seg = x[:, C.OFFSET_WAREHOUSE : C.OFFSET_MAP].view(-1, C.K_WAREHOUSE, C.WAREHOUSE_FEAT)
        map_seg = x[:, C.OFFSET_MAP : C.OFFSET_MAP + C.LOCAL_MAP_DIM]
        aux_seg = x[:, C.OFFSET_ACTION_AUX : C.OFFSET_ACTION_AUX + C.ACTION_AUX_DIM]
        return self_seg, station_seg, npc_seg, charger_seg, warehouse_seg, map_seg, aux_seg

    # ---------------- forward ---------------- #
    def forward(self, s, inference=False):
        s = s.to(torch.float32)
        (
            self_seg, station_seg, npc_seg, charger_seg, warehouse_seg,
            map_seg, aux_seg,
        ) = self._split(s)

        h_self = self.self_enc(self_seg)
        h_st = self.station_enc(station_seg)
        h_np = self.npc_enc(npc_seg)
        h_ch = self.charger_enc(charger_seg)
        h_wh = self.warehouse_enc(warehouse_seg)
        h_mp = self.map_enc(map_seg)
        h_aux = self.action_aux_enc(aux_seg)

        # 注意拼接顺序: 老特征段 (h_self..h_mp) 在前, aux 在后, 与 ckpt 兼容
        h = torch.cat([h_self, h_st, h_np, h_ch, h_wh, h_mp, h_aux], dim=-1)
        h = self.backbone(h)

        logits = self.actor_head(h)
        value = self.critic_head(h)
        return [logits, value]

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()

    # ------------------------------------------------------------------ #
    # 续训兼容加载: 支持老 ckpt (backbone.0.weight 是 352 输入) 的部分加载   #
    # ------------------------------------------------------------------ #
    def adapt_load_state_dict(self, state_dict, logger=None):
        """专用于加载老 ckpt 的方法.

        行为:
        1) 所有 shape 完全匹配的 key, 按正常 strict=True 流程加载.
        2) `backbone.0.weight`: 老 shape (128, 352), 新 shape (128, 384).
           把老权重拷到新权重的 [:, :352], 保持 [:, 352:] 的 0 初始化.
        3) `backbone.0.bias` / 其他 shape 匹配的老 key 正常加载.
        4) `action_aux_enc.*` 不在老 ckpt 中, 保持当前随机初始化.

        返回: dict, 描述加载情况.
        """
        own_state = self.state_dict()
        loaded = []
        skipped_shape_mismatch = []
        missing_in_ckpt = []

        for own_key, own_tensor in own_state.items():
            if own_key in state_dict:
                src_tensor = state_dict[own_key]
                if src_tensor.shape == own_tensor.shape:
                    own_tensor.copy_(src_tensor)
                    loaded.append(own_key)
                else:
                    # 特殊处理 backbone.0.weight
                    if (
                        own_key == "backbone.0.weight"
                        and src_tensor.shape[0] == own_tensor.shape[0]
                        and src_tensor.shape[1] <= own_tensor.shape[1]
                    ):
                        # 前 src_in 列拷过来, 其余保持 0 初始化
                        src_in = src_tensor.shape[1]
                        own_tensor[:, :src_in].copy_(src_tensor)
                        # 余下列保持 0 (不覆盖, 是 __init__ 里 zero_ 的)
                        loaded.append(f"{own_key} (partial: [:{src_in}])")
                    else:
                        skipped_shape_mismatch.append(
                            f"{own_key}: ckpt{tuple(src_tensor.shape)} vs model{tuple(own_tensor.shape)}"
                        )
            else:
                missing_in_ckpt.append(own_key)

        # ckpt 里但 model 里没有的 key (理论不该有, 除非结构改了)
        unexpected_in_ckpt = [k for k in state_dict if k not in own_state]

        report = {
            "loaded_count": len(loaded),
            "skipped_shape_mismatch": skipped_shape_mismatch,
            "missing_in_ckpt": missing_in_ckpt,  # 通常是 action_aux_enc.*
            "unexpected_in_ckpt": unexpected_in_ckpt,
        }
        if logger:
            logger.info(f"[adapt_load] loaded {len(loaded)} keys, "
                        f"missing(new) {len(missing_in_ckpt)}, "
                        f"skipped {len(skipped_shape_mismatch)}, "
                        f"unexpected {len(unexpected_in_ckpt)}")
            if skipped_shape_mismatch:
                logger.warning(f"[adapt_load] shape mismatch skipped: {skipped_shape_mismatch}")
        return report
