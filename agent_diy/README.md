# DIY 智运无人机 - 第一版设计说明

## 一、设计目标

基于用户在其他机器上训练的经验, 本版本针对三大痛点专门设计:

| 痛点 | 应对策略 |
| --- | --- |
| **耗电** (大量局死于电量耗尽) | 充电桩特征显式化 + 电量比例奖励 + 低电远桩惩罚 |
| **躲避 NPC 能力弱** | NPC 实体 slot + 威胁系数 + 近身阶梯惩罚 |
| **卡在角落** | 动作历史 + 位置抖动 + bounce 检测, 反抖动奖励 |

另外, 训练每次只能用单个 `train_env_conf.toml` 配置,
但测评环境的 `battery_max / drone_count / station_count / charger_count / max_step`
都会变. 为了让单次训练的模型能迁移到各种配置, 采用了一整套"**配置鲁棒**"设计 (下面详述).

## 二、配置鲁棒性: 换一组参数不用重训

这是第一版最核心的设计理念. 具体做法:

### 1. 所有归一化用"**比例 / 相对值**", 不用绝对值
- 电量: 只进 `battery / battery_max`, 不进绝对电量.
- 步数: 只进 `step_no / max_step`, 不进绝对步数.
- 投递数 / 充电次数: 归到一个大上界, 跨配置下分布相近.

### 2. 实体用"**定长 slot + found 位**"格式
| 实体 | 训练槽数 K | 每槽维度 |
| --- | --- | --- |
| 驿站 | 10 (站数上限) | 8 |
| NPC | 4 (NPC 数上限) | 6 |
| 充电桩 | 4 (桩数上限) | 6 |
| 仓库 | 1 | 6 |

测评时配置若比训练低, 剩余 slot 填 0 + `found=0`,
网络通过 `found` mask 自动忽略.

### 3. 把"**本局环境 meta**"作为特征送进网络
```
env_meta = [battery_max/999, max_step/2000, npc_cnt/4, station_cnt/10, charger_cnt/4]
```
网络可以根据"本局难度"自适应策略 —
例如 `charger_cnt=1` 时更谨慎用电.

## 三、特征向量 (共 234 维)

| 段 | 维度 | 内容 |
| --- | --- | --- |
| self_state | 10 | batt_ratio, batt_low, pkg_ratio, has_pkg, step_ratio, pos_x/y, delivered, charge_cnt, back_warehouse_cnt |
| action_history | 32 | 最近 4 步动作 one-hot (4×8) |
| stuck_detector | 4 | 位置方差, 重复比, 访问量, bounce 标志 |
| env_meta | 5 | battery_max/max_step/npc/station/charger 比例 |
| station slots | 80 | 10 × [found, dx, dy, ax, ay, dist, is_target, cfg_id] |
| npc slots | 24 | 4 × [found, dx, dy, dist, threat, approach] |
| charger slots | 24 | 4 × [found, dx, dy, dist, need_charge, priority] |
| warehouse slots | 6 | 1 × [found, dx, dy, dist, need_resupply, pad] |
| local_map | 49 | 21×21 视野 max-pool 成 7×7 |
| **合计** | **234** | |

驿站排序: **目标驿站优先**, 然后按距离升序.
NPC / 充电桩: 纯按距离升序.

## 四、奖励设计

| 奖励项 | 值 | 触发 |
| --- | --- | --- |
| 投递成功 | +1.0 / 个 | `delivered` 增加时 |
| 抵达仓库补货 | +0.05 | `back_warehouse_count` 增加 |
| 低电充电 | +0.03 × (1 - batt_ratio) | 电量 < 50% 进桩 |
| 向目标靠近 (势函数) | +0.02 × Δdist | 最近目标驿站距离缩短 |
| 探索新格子 | +0.005 | 到达新坐标 |
| 步数惩罚 | -0.0005 | 每步 (故意小) |
| 低电远桩 | -0.01 | 电量 < 20% 且离最近桩 > 6 格 |
| NPC 接近 | -0.05 | 最近 NPC 距离 ≤ 3 |
| NPC 临界 | -0.2 | 最近 NPC 距离 ≤ 1.5 |
| 卡角抖动 | -0.01 | repeat_ratio > 0.6 且方差小 |
| 撞墙 (bounce) | -0.02 | 执行动作但位置未变 |
| 局末异常终止 | -1.0 | terminated (碰撞/耗电) |
| 局末 0 投递 | -0.3 | truncated 且 delivered=0 |

### 为什么**不**直接做"耗电惩罚"?
直接按 Δbattery 扣奖励, 模型会学成"不动就不耗电" ->
更容易卡角. 所以改为: **低电远桩才扣**, **被动诱导回桩** 而不是"禁止耗电".

### 为什么 step 惩罚这么小?
在有"势函数 + 投递奖励"的情况下, 再叠加 -0.001 级别的 step 惩罚
足够引导效率; 过大会让策略不敢绕远路避 NPC, 反而撞死.

## 五、网络结构 (≈50K 参数, CPU 友好)

```
                 self(51) -> MLP -> 64
                                          \
 station(10,8)  -> SharedMLP -> pool -> 64  \
 npc(4,6)       -> SharedMLP -> pool -> 64   \
 charger(4,6)   -> SharedMLP -> pool -> 64    +--concat(352)--> MLP(128,128)
 warehouse(1,6) -> SharedMLP -> pool -> 64   /                    |   |
                                            /                   Actor Critic
 map(49)        -> MLP -> 32 ______________/                     (8)   (1)
```

### 关键点
- 每类实体用**共享 MLP** 处理每个 slot, 再做 **masked mean-pool + max-pool**.
  - mean-pool 表示"整体形势", max-pool 表示"最紧迫对象 (最近 NPC / 最紧迫目标)".
  - 用 slot 的 `found` 位做 mask, 空 slot 自动剔除.
- 实体数量不同时网络结构不变, **完美应对配置变化**.
- 局部地图用 MLP 而不是 CNN: 输入已是 49 维小向量, CNN 反而引入无意义的初始化噪声, 且 CPU 更慢.

## 六、训练超参 (on-policy PPO)

| 超参 | 值 | 说明 |
| --- | --- | --- |
| GAMMA | 0.995 | episode 长, 远视 |
| LAMBDA | 0.95 | GAE |
| START_LR | 3e-4 | Adam 默认 |
| CLIP_PARAM | 0.2 | 标准 PPO clip |
| VF_COEF | 0.5 | 价值头权重 |
| ENTROPY | 0.01 | 较大, 帮助摆脱卡角 |
| GRAD_CLIP | 0.5 | 防爆炸 |
| train_batch_size | 512 | CPU 可承受 |
| replay_buffer_capacity | 8192 | on-policy roll-out |

## 七、本地 13 代 i7 CPU 训练的加速细节

1. `torch.set_num_threads(1)` + `torch.set_num_interop_threads(1)`
   — 避免 torch 在多 worker 下互相抢线程导致吞吐反而下降.
2. on-policy PPO + `dump_model_freq=1` —
   每一轮训练都 dump, on-policy 要求 fresh.
3. 特征处理全部用 numpy, **不走 GPU**, 不做无用 device 搬运.
4. 网络参数 ~50K, i7 可以在 1-2ms 内完成一次前向.
5. 局部地图用 max-pool 预处理 (21×21 → 7×7), 不过 CNN.

## 八、训练环境配置 (第一版)

见 `agent_diy/conf/train_env_conf.toml`. 关键选择:

```
map_random = true          # 10 图随机, 泛化
drone_count = 4            # NPC 拉满, 从 day-1 就学躲
charger_count = 2          # 桩偏少, 强迫学电量管理
station_count = 10         # 目标丰富
battery_max = 300          # 默认, 网络用 ratio, 换了也能用
max_step = 1000            # 默认
```

### 想切其他难度?
直接改 `train_env_conf.toml` 重启训练. 由于特征全部归一化 + env_meta 写进特征,
**已有的 ckpt 可以直接 `preload_model=true` 继续训练**, 不需要重新从头.

## 九、监控面板（4 组）

**1. 结果占比（result_ratio, 单位 %）** — 直接反映失败原因:
- `fail_rate` / `fail_npc_rate` / `fail_battery_rate` / `fail_other_rate`
- `win_rate` / `win_nodeliver_rate`

**2. 平均表现（performance）** — 滑动窗口 100 局:
- `avg_reward` / `avg_delivered` / `avg_steps` / `avg_batt_left_ratio`
- 本局瞬时: `reward` / `delivered` / `battery_left`

**3. 痛点诊断（diagnostic）** — 直接对应三大痛点:
- `avg_bounce_per_ep`   — 撞墙次数(反映卡角)
- `avg_stuck_per_ep`    — 被判卡住的步数
- `avg_min_npc_dist`    — 局内 NPC 最近距离的均值(越大越安全)
- `avg_low_batt_ratio`  — 本局处于低电量(<20%)的步数占比

**4. 算法（algorithm）** — total_loss / value_loss / policy_loss / entropy_loss.

### 局末原因推断逻辑
环境 `terminated` 不给具体失败码, 在 workflow 的 `classify_result()` 里按最后一帧推断:
1. `truncated` → `WIN` / `WIN_NODELIVER` (看 delivered)
2. `terminated & battery <= 0` → `FAIL_BATTERY`
3. `terminated & last_npc_dist <= 2.0` → `FAIL_NPC`
4. 其他 `terminated` → `FAIL_OTHER`

滑动窗口大小 100, 每 60s 上报一次, 既能快速反映最近训练状态, 也不会被单局噪声带偏.

## 十、后续迭代方向 (v2)

1. **LSTM / GRU** 跨步记忆: 当前只用了 4 步动作历史, 时序建模可以提升 NPC 追踪.
2. **双 critic**: 把"生存价值 (避免 terminated)"和"任务价值 (投递)"拆开, 分别 GAE, 减少梯度冲突.
3. **动作 soft mask**: 除了 `legal_action`, 再用 NPC 方向做软 logit 扣减 (而不是硬 mask, 避免死路).
4. **RND 探索**: 用随机网络蒸馏加强新地图适应.
5. **行为克隆 warm-up**: 写一个简单的 A* + 贪心 NPC 回避的脚本, 生成少量专家数据预训练 actor, 再 PPO.

## 十、文件清单

```
agent_diy/
├── agent.py                         # Agent 主类, predict / learn / save / load
├── conf/
│   ├── conf.py                      # 全部超参 + 特征维度
│   ├── train_env_conf.toml          # 训练环境配置
│   └── monitor_builder.py           # 监控面板
├── feature/
│   ├── definition.py                # ObsData / ActData / SampleData, GAE
│   └── preprocessor.py              # 特征 + 奖励核心实现
├── model/model.py                   # 分组编码器 + 主干
├── algorithm/algorithm.py           # PPO clip loss
└── workflow/train_workflow.py       # episode runner
```

全局配置:
- `conf/configure_app.toml`  → train_batch_size=512, on-policy
- `conf/algo_conf_drone_delivery.toml` / `conf/app_conf_drone_delivery.toml` → algo=diy
- `train_test.py` → algorithm_name="diy"
