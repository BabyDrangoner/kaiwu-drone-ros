-- ============================================================================
-- drone - 小悟无人机（地图编辑器内运行时联调版本）
-- 路径: scripts/gameplay/drone_delivery/main.lua
-- 参考: scripts/gameplay/gorge_chase/main.lua, scripts/gameplay/robot_vacuum/main.lua
-- ============================================================================
-- 注意：运行时 Lua 环境是沙箱，dofile/loadfile 被禁用，请使用 require
--
-- 需求摘要（本脚本覆盖）：
-- 1) 新增元素高度属性：使用 cell.height（地形） + 对象 properties.height（设施/建筑）
-- 2) 无人机可飞越一定高度的元素：高度 <= fly_over_height 可通过；更高则不可
-- 3) 不可行走区域一直不可行走：map.is_walkable=false 一律不可通过（无视飞越）
-- 4) 包裹/仓库/企鹅驿站/充电桩：仓库补给3包裹；驿站投递+100；充电桩范围内瞬充
-- ============================================================================

-- 键盘调试控制（数字键 0-7）
local Control = require("control.control")
local GridUnit = require("ecs.grid_unit")
local ECS = require("ecs.ecs")
local ECSCommon = require("ecs.common")
local DronePlayerElement = require("ecs.elements.drone_player")
local EnemyElement = require("ecs.elements.enemy_spawn")
local StationElement = require("ecs.elements.station")
local ChargerElement = require("ecs.elements.charger")
local WarehouseElement = require("ecs.elements.warehouse")
local BaseObjectElement = require("ecs.elements.base_object")
local RobotProfiles = require("ecs.robot_profiles")
local DroneDeliveryFrame = require("gameplay.drone_delivery.frame")
local DroneDeliveryStatus = require("gameplay.drone_delivery.status")
local DroneDeliveryWorld = require("gameplay.drone_delivery.world")

-- ============================================================================
-- 默认配置（Default Configuration）
-- ============================================================================
local function load_default_config_from_env()
    -- 由 Rust 侧将 config/drone_delivery/env_config.toml 注入到 Lua 全局
    local injected = _G.__drone_user_config
    if type(injected) == "table" then
        return ECSCommon.deep_copy(injected)
    end
    return {}
end

local DEFAULT_CONFIG = load_default_config_from_env()

-- 无人机颜色：空载/携带（用于"拿到东西变色，放下还原"）
-- 说明：避免与点位/道具（常用金色/紫色/黄）冲突，选用更偏"青绿/粉紫"两组。
local DRONE_COLOR_EMPTY = { 0, 220, 120, 255 }
local DRONE_COLOR_CARRY = { 255, 80, 200, 255 }


-- ============================================================================
-- 常量与全局状态
-- ============================================================================


-- 方向编号规则（0-7）：0=右，1=右上，2=上，3=左上，4=左，5=左下，6=下，7=右下
local DIRECTION_VECTORS = {
    [0] = { x = 1, z = 0 },
    [1] = { x = 1, z = -1 },
    [2] = { x = 0, z = -1 },
    [3] = { x = -1, z = -1 },
    [4] = { x = -1, z = 0 },
    [5] = { x = -1, z = 1 },
    [6] = { x = 0, z = 1 },
    [7] = { x = 1, z = 1 },
}

local deep_copy = ECSCommon.deep_copy
local merge_config = ECSCommon.merge_config
local collect_sorted_keys = ECSCommon.collect_sorted_keys
local key_diff_report = ECSCommon.key_diff_report

local clamp = ECSCommon.clamp

local random_choice = ECSCommon.random_choice

local shuffle_inplace = ECSCommon.shuffle_inplace
local key_xy = ECSCommon.xy_key
local merge_tags = ECSCommon.merge_tags

-- ============================================================================
local function parse_number(v, default)
    if v == nil then return default end
    local n = tonumber(v)
    if n == nil then return default end
    return n
end

-- 游戏状态（全局，便于回调/调试）
game_state = {
    -- 实体
    player = nil,
    player_id = nil,

    -- 地图
    map_width = 0,
    map_height = 0,
    current_map_id = 1,

    -- 任务状态
    current_step = 0,
    max_step = 1000,
    task_status = "running", -- running/completed/failed
    fail_reason = "",

    -- 玩法数据
    score = 0,
    delivered = 0,
    last_step_reward = 0,
    battery = 100,
    battery_max = 100,
    player_fly_over_bonus = 0.0,
    player_battery_cost_per_step = 1,

    packages = {},        -- {station_id, ...}

    stations = {},        -- { {id, x, z, entity_id}, ... }
    station_cells = {},   -- key -> station_id（N×N 逻辑由多个 1x1 单元组成）

    chargers = {},        -- { {x, z, range, entity_id}, ... }
    charger_cells = {},   -- key -> true（N×N 逻辑由多个 1x1 单元组成）
    charger_preset_anchor_cache = {}, -- map_id -> full preset charger anchors snapshot
    warehouse_cells = {}, -- key -> true（真实 warehouse 对象占地，供补货/进入判定使用）
    station_forbidden_cells = {}, -- key -> true（station/charger 生成禁放区域：warehouse + warehouse_ui）
    warehouses = {},      -- 新增：仓库区域列表 { {x, z, w, h}, ... }
    warehouse_return_count = 1, -- 返回仓库次数（起始算一次）

    charger_charge_count = 0, -- 补电次数（进入充电桩范围或进入仓库计一次）

    -- 官方无人机（巡逻用，不影响玩家步数/能量）
    official_drones = {}, -- { {entity_id, home_x, home_z, radius, accum}, ... }

    -- 进入判定（用于提示节流；效果本身每步都可触发）
    in_warehouse = false,
    in_charger = false,
    last_station_entered = nil,


    -- 高度缓存（对象高度映射到格子）
    obj_height_by_cell = {}, -- key -> max_height

    -- 调试面板/可视化状态（供 Rust 调试地图面板读取）
    map_viz_visible = false,
    last_map_viz_grid = nil,
    last_map_viz_cx = nil,
    last_map_viz_cz = nil,
    last_map_viz_vr = nil,
}


local usr_config = {}

local function is_in_bounds(x, z)
    return ECSCommon.is_in_bounds(x, z, game_state.map_width, game_state.map_height)
end

-- 不可行走区域一直不可行走：底层以 map.is_walkable 为准
local function is_base_walkable(x, z)
    if not is_in_bounds(x, z) then
        return false
    end
    return map.is_walkable(x, z)
end

local _passability_api = nil

local function ensure_passability_api()
    if _passability_api then
        return _passability_api
    end

    _passability_api = BaseObjectElement.new_passability_api({
        map_api = map,
        key_xy_fn = key_xy,
        is_in_bounds_fn = is_in_bounds,
        is_base_walkable_fn = is_base_walkable,
        get_obj_height_cache_fn = function()
            return game_state.obj_height_by_cell
        end,
        obstacle_epsilon = 0.01,
    })

    return _passability_api
end

local function is_passable_for_drone(x, z)
    return ensure_passability_api().is_passable(
        x,
        z,
        usr_config.fly_over_height or 2.0,
        game_state.player_fly_over_bonus or 0.0
    )
end


--- 检查斜向移动是否可行（不允许"卡角"穿墙）
local function can_move_diagonally(from_x, from_z, to_x, to_z)
    return ECSCommon.can_move_diagonally(from_x, from_z, to_x, to_z, is_passable_for_drone)
end

-- ============================================================================
-- 官方无人机（NPC）出生点与巡逻
-- ============================================================================

local CELL_ROAD = 3
local CELL_BRIDGE = 4
local CELL_GRASS = 5
local CELL_WAREHOUSE_UI = 8

local function is_road_like_cell(x, z)
    if not map.get_cell_type then
        -- 没有 cell_type 时退化为"只要可通行就当作道路类"
        return is_base_walkable(x, z)
    end
    local t = map.get_cell_type(x, z) or 0
    return (t == CELL_ROAD) or (t == CELL_BRIDGE)
end

--- 判断 (x, y) 是否为"道路或草地"类型（官方无人机允许生成的地形）
local function is_road_or_grass(x, z)
    if not map.get_cell_type then
        -- cell_type 不可用时，退化为 walkable
        return is_base_walkable(x, z)
    end
    local t = map.get_cell_type(x, z) or 0
    return (t == CELL_ROAD) or (t == CELL_GRASS)
end

--- 判断格子 (x, y) 是否在任一仓库区域扩展 buffer 格后的范围内
local function is_in_warehouse_buffer(x, z, buffer)
    local whs = game_state.warehouses
    if not whs then return false end
    for _, wh in ipairs(whs) do
        if x >= wh.x - buffer and x < wh.x + wh.w + buffer
            and z >= wh.z - buffer and z < wh.z + wh.h + buffer then
            return true
        end
    end
    return false
end

local function get_official_drone_spawn_buffer(radius)
    local r = math.max(0, math.floor((tonumber(radius) or 0) + 1e-6))
    return math.max(8, r + 1)
end

local function is_valid_official_drone_spawn_cell(x, z, radius)
    return is_in_bounds(x, z)
        and is_road_or_grass(x, z)
        and is_passable_for_drone(x, z)
        and not is_in_warehouse_buffer(x, z, get_official_drone_spawn_buffer(radius))
end

local function is_valid_official_drone_patrol_cell(x, z)
    return is_in_bounds(x, z)
        and is_passable_for_drone(x, z)
        and not is_in_warehouse_buffer(x, z, 0)
end

local function find_safe_official_drone_cell(origin_x, origin_z, radius)
    if is_valid_official_drone_spawn_cell(origin_x, origin_z, radius) then
        return { x = origin_x, z = origin_z }
    end

    local max_search_radius = math.max(12, math.floor((tonumber(radius) or 0) + 12))
    local best = nil
    local best_dist2 = nil

    for search_radius = 1, max_search_radius do
        for dx = -search_radius, search_radius do
            for dz = -search_radius, search_radius do
                if math.abs(dx) ~= search_radius and math.abs(dz) ~= search_radius then
                    goto continue
                end

                local x = origin_x + dx
                local z = origin_z + dz
                if is_valid_official_drone_spawn_cell(x, z, radius) then
                    local dist2 = dx * dx + dz * dz
                    if best == nil or dist2 < best_dist2 then
                        best = { x = x, z = z }
                        best_dist2 = dist2
                    end
                end

                ::continue::
            end
        end

        if best then
            return best
        end
    end

    return nil
end

local function spawn_official_drones()
    game_state.official_drones = {}
    local profile = RobotProfiles.get("drone_delivery", "official") or {}

    -- 获取配置的无人机数量（默认 4，范围 0-10）
    local drone_count = clamp(usr_config and usr_config.drone_count or 4, 0, 10)
    if drone_count <= 0 then
        log("[OfficialDrone] drone_count=0, skip spawning")
        return
    end

    local points = EnemyElement.query_drone_official_spawn_points(ECS)
    if not points or #points == 0 then
        -- 说明：部分导出地图只有 cells，没有放置官方无人机出生点对象。
        -- 兜底策略：扫描地图可通行格子，随机选取若干个尽量分散的点位生成。
        log_warn("[OfficialDrone] No official spawn objects found; generating fallback spawn points")

        local function pick_fallback_points(count)
            local w = game_state.map_width or 0
            local h = game_state.map_height or 0
            if w <= 0 or h <= 0 or count <= 0 then
                return {}
            end

            local candidates = {}

            -- 大地图采样优化
            local sample_step = 1
            if w * h > 10000 then
                sample_step = 2
            end

            for x = 0, w - 1, sample_step do
                for z = 0, h - 1, sample_step do
                    if is_valid_official_drone_spawn_cell(x, z, 10) then
                        table.insert(candidates, { x = x, z = z })
                    end
                end
            end

            if #candidates == 0 then
                return {}
            end

            shuffle_inplace(candidates)

            -- iWiki：活动范围 21x21（半径 10）。这里用一个偏保守的最小间距避免明显重叠。
            local min_dist2 = 25 * 25
            local selected = {}

            for _, c in ipairs(candidates) do
                local ok = true
                for _, s in ipairs(selected) do
                    local dx = c.x - s.x
                    local dz = c.z - s.z
                    if (dx * dx + dz * dz) < min_dist2 then
                        ok = false
                        break
                    end
                end
                if ok then
                    table.insert(selected, c)
                    if #selected >= count then
                        break
                    end
                end
            end

            return selected
        end

        local picked = pick_fallback_points(drone_count)
        if not picked or #picked == 0 then
            log_warn("[OfficialDrone] No passable candidates for fallback spawning, skip")
            return
        end

        points = {}
        for _, p in ipairs(picked) do
            table.insert(points, { x = p.x, z = p.z, properties = { patrol_radius = 10 } })
        end
    end

    -- 随机打乱出生点顺序，然后尽量生成到目标数量
    shuffle_inplace(points)
    local target_spawn_count = math.min(drone_count, #points)
    log(string.format("[OfficialDrone] Spawning up to %d/%d drones (points=%d)", target_spawn_count, drone_count, #points))

    local occupied_spawn = {}
    local actual_spawned = 0

    for _, p in ipairs(points) do
        if actual_spawned >= target_spawn_count then
            break
        end

        local radius = 10
        if p.properties and p.properties.patrol_radius then
            radius = clamp(parse_number(p.properties.patrol_radius, 10), 1, 50)
        end
        local radius_scale = tonumber(profile.patrol_radius_scale) or 1.0
        radius = clamp(math.floor(radius * radius_scale + 1e-6), 1, 50)

        local safe_spawn = find_safe_official_drone_cell(p.x, p.z, radius)
        if not safe_spawn then
            log_warn(string.format(
                "[OfficialDrone] Skip unsafe spawn point at (%d,%d): no safe road/grass cell outside warehouse buffer",
                p.x,
                p.z
            ))
            goto continue
        end

        local spawn_x = safe_spawn.x
        local spawn_z = safe_spawn.z
        if spawn_x ~= p.x or spawn_z ~= p.z then
            log_warn(string.format(
                "[OfficialDrone] Relocated spawn from (%d,%d) to (%d,%d) to avoid warehouse area",
                p.x,
                p.z,
                spawn_x,
                spawn_z
            ))
        end

        local spawn_key = key_xy(spawn_x, spawn_z)
        if occupied_spawn[spawn_key] then
            log_warn(string.format(
                "[OfficialDrone] Skip overlapped spawn point at (%d,%d)",
                spawn_x,
                spawn_z
            ))
            goto continue
        end

        local eid = entity.spawn(
            profile.tag or "drone_official",
            spawn_x + 0.5,
            spawn_z + 0.5,
            merge_tags({ "enemy", "drone", "official" }, profile.tags),
            {
                collision = {
                    enabled = false,
                    layer = 1,
                    mask = 0,
                    shape = {
                        type = "circle",
                        radius = tonumber(profile.collision_size and profile.collision_size.radius) or 0.35,
                    },
                },
                visual = {
                    display_name = profile.name or "官方无人机",
                    color = profile.color or { 80, 160, 255, 255 },
                    visible = true,
                },
            }
        )

        if eid then
            occupied_spawn[spawn_key] = eid
            actual_spawned = actual_spawned + 1
            entity.set_property(eid, "robot_profile", profile.class_name or "standard")
            EnemyElement.consume_source_instance(p, "drone_delivery_official")
            table.insert(game_state.official_drones,
                {
                    entity_id = eid,
                    home_x = spawn_x,
                    home_z = spawn_z,
                    radius = radius,
                    accum = 0.0,
                    step_interval = tonumber(profile.patrol_step_interval) or 0.35,
                })
        end

        ::continue::
    end

    if actual_spawned < target_spawn_count then
        log_warn(string.format(
            "[OfficialDrone] Spawned %d/%d drones after applying warehouse safety constraints",
            actual_spawned,
            target_spawn_count
        ))
    end
end

local function update_official_drones(dt)
    if not game_state.official_drones or #game_state.official_drones == 0 then
        return
    end

    -- 规则对齐 gorge_chase：官方无人机按“步”移动，而非按时间移动。
    -- 即：玩家每执行一次动作，官方无人机移动一次。
    -- 这里保留 dt 参数仅为兼容调用签名。

    -- 仅 4 方向随机巡逻
    local dirs = {
        { 1,  0 },
        { -1, 0 },
        { 0,  1 },
        { 0,  -1 },
    }

    local occupied = {}
    for _, d in ipairs(game_state.official_drones) do
        local e = entity.get(d.entity_id)
        if e then
            occupied[key_xy(math.floor(e.x), math.floor(e.z))] = d.entity_id
        end
    end

    for _, d in ipairs(game_state.official_drones) do
        local e = entity.get(d.entity_id)
        if not e then
            goto continue
        end

        local gx = math.floor(e.x)
        local gz = math.floor(e.z)
        local current_key = key_xy(gx, gz)
        occupied[current_key] = d.entity_id

        -- 尝试若干次随机方向，活动范围：出生点 ±radius 格（默认 radius=10，即 21×21）
        for _ = 1, 6 do
            local dv = random_choice(dirs)
            local tx = gx + dv[1]
            local tz = gz + dv[2]
            local target_key = key_xy(tx, tz)
            local occupied_by = occupied[target_key]

            if is_valid_official_drone_patrol_cell(tx, tz)
                and math.abs(tx - d.home_x) <= d.radius
                and math.abs(tz - d.home_z) <= d.radius
                and (occupied_by == nil or occupied_by == d.entity_id)
            then
                entity.move_to(d.entity_id, tx + 0.5, tz + 0.5)
                occupied[current_key] = nil
                occupied[target_key] = d.entity_id
                break
            end
        end

        ::continue::
    end
end

-- ============================================================================
-- 碰撞判定：靠近怪物 3*3 即失败
-- ============================================================================

local function check_collision_player_vs_enemies()
    return EnemyElement.fail_if_enemy_near_player(game_state, {
        range = 1,
        fail_reason = "碰到怪物",
        toast_message = "碰到怪物，任务失败",
        toast_duration = 2.0,
        require_running = true,
    })
end

local function rebuild_object_height_cache()
    game_state.obj_height_by_cell = BaseObjectElement.build_height_cache_from_map(map, key_xy, is_in_bounds)
end

local function init_warehouse_cells()
    game_state.warehouse_cells = {}
    game_state.station_forbidden_cells = {}
    game_state.warehouses = {}  -- 新增：缓存仓库区域信息

    local function mark_station_forbidden_cell(x, z)
        if is_in_bounds(x, z) then
            game_state.station_forbidden_cells[key_xy(x, z)] = true
        end
    end

    local warehouses = WarehouseElement.query_warehouse_areas(ECS)

    -- 去重：同一左上角可能存在重复对象（例如编辑器误放多次），避免重复处理
    local seen_top_left = {}

    for _, obj in ipairs(warehouses or {}) do
        local ox = obj.x
        local oz = obj.z
        local w = obj.w
        local h = obj.h

        local k0 = key_xy(ox, oz)
        if not seen_top_left[k0] then
            seen_top_left[k0] = true

            -- 缓存仓库区域信息（全局绝对位置）
            table.insert(game_state.warehouses, { x = ox, z = oz, w = w, h = h })

            GridUnit.for_each_rect_cell(ox, oz, w, h, function(x, z)
                if is_in_bounds(x, z) then
                    local k = key_xy(x, z)
                    game_state.warehouse_cells[k] = true
                    game_state.station_forbidden_cells[k] = true
                end
            end)
        end
    end

    if map.get_cell_type then
        local w = game_state.map_width or 0
        local h = game_state.map_height or 0
        for x = 0, w - 1 do
            for z = 0, h - 1 do
                if (map.get_cell_type(x, z) or 0) == CELL_WAREHOUSE_UI then
                    mark_station_forbidden_cell(x, z)
                end
            end
        end
    end

    -- 不再额外 spawn 仓库可视化实体：地图物体层已经能显示仓库；
    -- 额外实体会在画布上叠很多圆圈/朝向点，影响观察。
end


local function sync_drone_visual()
    if not game_state.player_id then
        return
    end

    local carrying = (game_state.packages ~= nil and #game_state.packages > 0)
    local c = carrying and DRONE_COLOR_CARRY or DRONE_COLOR_EMPTY

    -- 依赖 Rust 侧提供 entity.set_color；若缺失则提示一次（方便定位"为什么不变色"）
    if not entity.set_color then
        if not game_state._warned_no_set_color then
            game_state._warned_no_set_color = true
            log_warn("[Visual] entity.set_color not available; drone color will not change")
        end
        return
    end

    local ok = entity.set_color(game_state.player_id, c[1], c[2], c[3], c[4])
    if ok == false then
        log_warn("[Visual] set_color failed")
    end
end


local function pick_station_id_for_package()
    if not game_state.stations or #game_state.stations == 0 then
        return nil
    end

    local used = {}
    for _, sid in ipairs(game_state.packages or {}) do
        used[sid] = true
    end

    local candidates = {}
    for _, s in ipairs(game_state.stations) do
        if not used[s.id] then
            table.insert(candidates, s.id)
        end
    end

    if #candidates == 0 then
        return random_choice(game_state.stations).id
    end

    return random_choice(candidates)
end

-- ============================================================================
-- Voronoi 风格均匀点选取（用于驿站自动生成）
-- ============================================================================

local STATION_MIN_DISTANCE = 5 -- Minimum spacing between station anchors (top-left)
local STATION_SIZE = 3         -- Station footprint size is 3x3, composed from 1x1 visual units
local STATION_EDGE_BUFFER = 5  -- Minimum gap from map border for station footprint
local STATION_WAREHOUSE_BUFFER = 2 -- Minimum gap from warehouse footprint for station placement
local CHARGER_UNIT_OBJECT_NAME = "charger_1x1"
local STATION_UNIT_OBJECT_NAME = "penguin_station_point"

-- Check whether a top-left anchored rectangle is fully passable
local function is_rect_passable(ox, oz, w, h, prefer_road)
    local rw = math.max(1, math.floor((w or STATION_SIZE) + 1e-6))
    local rh = math.max(1, math.floor((h or STATION_SIZE) + 1e-6))

    for x = ox, ox + rw - 1 do
        for z = oz, oz + rh - 1 do
            if not is_in_bounds(x, z) then
                return false
            end
            if not is_passable_for_drone(x, z) then
                return false
            end
            if prefer_road == true and not is_road_like_cell(x, z) then
                return false
            end
        end
    end

    return true
end

-- Check whether a top-left anchored rectangle overlaps or is too close to station-forbidden cells
local function is_rect_overlapping_warehouse(ox, oz, w, h, buffer)
    if not game_state.station_forbidden_cells then
        return false
    end

    local rw = math.max(1, math.floor((w or STATION_SIZE) + 1e-6))
    local rh = math.max(1, math.floor((h or STATION_SIZE) + 1e-6))
    local gap = math.max(0, math.floor((buffer or 0) + 1e-6))
    for x = ox - gap, ox + rw - 1 + gap do
        for z = oz - gap, oz + rh - 1 + gap do
            if game_state.station_forbidden_cells[key_xy(x, z)] then
                return true
            end
        end
    end
    return false
end

local function is_rect_too_close_to_map_edge(ox, oz, w, h, edge_buffer)
    local rw = math.max(1, math.floor((w or STATION_SIZE) + 1e-6))
    local rh = math.max(1, math.floor((h or STATION_SIZE) + 1e-6))
    local buffer = math.max(0, math.floor((edge_buffer or 0) + 1e-6))
    local min_x = buffer
    local min_z = buffer
    local max_x = (game_state.map_width or 0) - buffer - rw
    local max_z = (game_state.map_height or 0) - buffer - rh

    return ox < min_x or oz < min_z or ox > max_x or oz > max_z
end

-- Collect candidate top-left anchors (road-first) for a STATION_SIZE footprint
local function collect_anchor_candidates()
    local road_anchors = {}
    local other_anchors = {}

    local w = game_state.map_width or 0
    local h = game_state.map_height or 0

    -- Sampling optimization for large maps
    local sample_step = 1
    if w * h > 10000 then
        sample_step = 2
    end

    local max_x = w - STATION_SIZE
    local max_z = h - STATION_SIZE
    if max_x < 0 or max_z < 0 then
        return { road_anchors = road_anchors, other_anchors = other_anchors }
    end

    for x = 0, max_x, sample_step do
        for z = 0, max_z, sample_step do
            if is_rect_too_close_to_map_edge(x, z, STATION_SIZE, STATION_SIZE, STATION_EDGE_BUFFER) then
                goto continue
            end
            if is_rect_overlapping_warehouse(x, z, STATION_SIZE, STATION_SIZE, STATION_WAREHOUSE_BUFFER) then
                goto continue
            end
            if is_rect_passable(x, z, STATION_SIZE, STATION_SIZE, true) then
                table.insert(road_anchors, { x = x, z = z })
            elseif is_rect_passable(x, z, STATION_SIZE, STATION_SIZE, false) then
                table.insert(other_anchors, { x = x, z = z })
            end
            ::continue::
        end
    end

    return { road_anchors = road_anchors, other_anchors = other_anchors }
end

-- K-means++ 风格选取均匀分布的左上角锚点
-- @param anchor_candidates 候选锚点列表
-- @param count 需要选取的数量
-- @param min_anchor_dist 锚点最小间距
-- @param existing_anchors 已有锚点（需要避开）
-- @return 选中的锚点列表
local function select_spread_anchors_kmeans(anchor_candidates, count, min_anchor_dist, existing_anchors)
    local selected_anchors = ECSCommon.select_spread_points_kmeans(anchor_candidates, count, min_anchor_dist, existing_anchors)
    if #selected_anchors == 0 and anchor_candidates and #anchor_candidates > 0 and count > 0 then
        log_warn("[Station/Voronoi] No valid anchors after filtering by existing anchors")
    end
    return selected_anchors
end

-- ============================================================================
-- 驿站/充电桩点位获取
-- ============================================================================

local function format_anchor_list(anchors)
    if not anchors or #anchors == 0 then
        return "[]"
    end

    local parts = {}
    for i, p in ipairs(anchors) do
        parts[i] = string.format(
            "(%d,%d,%dx%d)",
            tonumber(p.x) or -1,
            tonumber(p.z) or -1,
            math.max(1, math.floor((tonumber(p.w) or STATION_SIZE) + 1e-6)),
            math.max(1, math.floor((tonumber(p.h) or STATION_SIZE) + 1e-6))
        )
    end
    return "[" .. table.concat(parts, ", ") .. "]"
end

local function clone_charger_anchor(anchor)
    return {
        x = tonumber(anchor.x) or 0,
        z = tonumber(anchor.z) or 0,
        w = math.max(1, math.floor((tonumber(anchor.w) or STATION_SIZE) + 1e-6)),
        h = math.max(1, math.floor((tonumber(anchor.h) or STATION_SIZE) + 1e-6)),
        range = anchor.range,
    }
end

local function clone_charger_anchor_list(anchors)
    local cloned = {}
    for _, anchor in ipairs(anchors or {}) do
        table.insert(cloned, clone_charger_anchor(anchor))
    end
    return cloned
end

local function get_map_key_for_charger_cache()
    return tostring(game_state.map_id or usr_config.map_id or usr_config.map_name or "default")
end

local function get_full_preset_charger_anchors()
    local map_key = get_map_key_for_charger_cache()
    local cached = game_state.charger_preset_anchor_cache[map_key]
    if cached and #cached > 0 then
        return clone_charger_anchor_list(cached), true
    end

    local queried = ChargerElement.query_drone_charger_points(ECS)
    local snapshot = clone_charger_anchor_list(queried)
    game_state.charger_preset_anchor_cache[map_key] = snapshot
    return clone_charger_anchor_list(snapshot), false
end

local function init_stations_and_chargers()
    game_state.stations = {}
    game_state.station_cells = {}
    game_state.chargers = {}
    game_state.charger_cells = {}

    -- 企鹅驿站：优先使用预设锚点，不足时用 Voronoi 补齐
    local preset_station_anchors = StationElement.query_drone_station_points(ECS)
    -- 不再 shuffle：保持与地图 JSON 定义的一致顺序，便于观测值与地图对应

    local desired_station = clamp(usr_config.station_count or 4, 1, 10)
    local picked_station_anchors = {}

    -- 1. 优先使用预设锚点（左上角锚点）
    for _, p in ipairs(preset_station_anchors) do
        if #picked_station_anchors >= desired_station then
            break
        end

        if is_rect_too_close_to_map_edge(p.x, p.z, STATION_SIZE, STATION_SIZE, STATION_EDGE_BUFFER) then
            log_warn(string.format("[Station] Preset anchor (%d,%d) is too close to map edge (buffer=%d), skipped", p.x, p.z, STATION_EDGE_BUFFER))
        elseif is_rect_overlapping_warehouse(p.x, p.z, STATION_SIZE, STATION_SIZE, STATION_WAREHOUSE_BUFFER) then
            log_warn(string.format("[Station] Preset anchor (%d,%d) is too close to warehouse / warehouse_ui area (buffer=%d), skipped", p.x, p.z, STATION_WAREHOUSE_BUFFER))
        elseif not is_rect_passable(p.x, p.z, STATION_SIZE, STATION_SIZE, false) then
            log_warn(string.format("[Station] Preset anchor (%d,%d) footprint is not passable, skipped", p.x, p.z))
        else
            table.insert(picked_station_anchors, {
                id = p.id,
                x = p.x,
                z = p.z,
            })
        end
    end

    -- 2. 锚点不足时，用 Voronoi 补齐
    local gap = desired_station - #picked_station_anchors
    if gap > 0 then
        log(string.format("[Station] Preset anchors=%d, need %d more, using Voronoi", #picked_station_anchors, gap))

        -- 收集候选左上角锚点（道路优先）
        local anchors = collect_anchor_candidates()
        local anchor_candidates = {}

        -- 道路锚点优先
        for _, anchor in ipairs(anchors.road_anchors) do
            table.insert(anchor_candidates, anchor)
        end
        -- 道路不足时加入其他可行锚点
        if #anchor_candidates < gap * 10 then
            for _, anchor in ipairs(anchors.other_anchors) do
                table.insert(anchor_candidates, anchor)
            end
        end

        if #anchor_candidates == 0 then
            log_error("[Station] No passable anchors for Voronoi generation")
        else
            -- 构建已有锚点集（全程保持左上角语义）
            local existing_anchors = {}
            for _, p in ipairs(picked_station_anchors) do
                table.insert(existing_anchors, {
                    x = p.x,
                    z = p.z,
                })
            end

            -- K-means++ 风格选取锚点
            local voronoi_anchors = select_spread_anchors_kmeans(anchor_candidates, gap, STATION_MIN_DISTANCE, existing_anchors)

            log(string.format("[Station] Voronoi generated %d anchors", #voronoi_anchors))

            for _, anchor in ipairs(voronoi_anchors) do
                table.insert(picked_station_anchors, { id = nil, x = anchor.x, z = anchor.z })
            end
        end
    end

    log(string.format("[Station] Total stations: %d (desired: %d)", #picked_station_anchors, desired_station))

    local picked_station_anchor_texts = {}
    for _, p in ipairs(picked_station_anchors) do
        table.insert(picked_station_anchor_texts, string.format("(%d,%d)", p.x, p.z))
    end
    log(string.format("[Station] Picked anchors=[%s]", table.concat(picked_station_anchor_texts, ", ")))

    -- station_id 去重 + 自动分配（保证 1..10 且尽量唯一）
    local used_ids = {}
    for _, p in ipairs(picked_station_anchors) do
        if p.id ~= nil and p.id >= 1 and p.id <= 10 and (not used_ids[p.id]) then
            used_ids[p.id] = true
        else
            p.id = nil
        end
    end

    local available_ids = {}
    for id = 1, 10 do
        if not used_ids[id] then
            table.insert(available_ids, id)
        end
    end

    local function alloc_station_id()
        if #available_ids == 0 then
            return nil
        end
        return table.remove(available_ids, 1)
    end

    for _, p in ipairs(picked_station_anchors) do
        if p.id == nil then
            p.id = alloc_station_id()
        end

        if p.id ~= nil then
            local spawn_center_x = p.x + math.floor(STATION_SIZE / 2)
            local spawn_center_z = p.z + math.floor(STATION_SIZE / 2)
            local center_eid, composed_cells = GridUnit.spawn_tiled_units({
                center_x = spawn_center_x,
                center_z = spawn_center_z,
                size = STATION_SIZE,
                unit_name = STATION_UNIT_OBJECT_NAME,
                tags = { "station", "penguin_station", STATION_UNIT_OBJECT_NAME },
                color = { 170, 90, 255, 255 },
                center_label = "驿站#" .. tostring(p.id),
                in_bounds_fn = is_in_bounds,
                key_fn = key_xy,
            })

            -- 3x3 驿站逻辑 = 多个 1x1 单元（直接复用 GridUnit 返回的 cell_map）
            for k, _ in pairs(composed_cells or {}) do
                game_state.station_cells[k] = p.id
            end

            table.insert(game_state.stations, {
                id = p.id,
                x = p.x,
                z = p.z,
                w = STATION_SIZE,
                h = STATION_SIZE,
                entity_id = center_eid,
            })
        end
    end


    -- 充电桩：从预设锚点中随机选取（同图多次 reset 也基于完整候选集重新随机）
    local preset_charger_anchors, preset_from_cache = get_full_preset_charger_anchors()
    local desired_charger = clamp(usr_config.charger_count or 2, 1, 4)
    log(string.format("[ChargerDebug] desired=%d, preset_count=%d, preset_from_cache=%s, preset_anchors=%s", desired_charger, #preset_charger_anchors, tostring(preset_from_cache), format_anchor_list(preset_charger_anchors)))

    -- 1. 筛出所有可通行的预设锚点
    local passable_charger_anchors = {}
    for _, p in ipairs(preset_charger_anchors) do
        local pw = math.max(1, math.floor((tonumber(p.w) or STATION_SIZE) + 1e-6))
        local ph = math.max(1, math.floor((tonumber(p.h) or STATION_SIZE) + 1e-6))
        if is_rect_passable(p.x, p.z, pw, ph, false) then
            table.insert(passable_charger_anchors, {
                x = p.x,
                z = p.z,
                w = pw,
                h = ph,
                range = p.range,
            })
        else
            log_warn(string.format("[Charger] Preset anchor (%d,%d) footprint is not passable, skipped", p.x, p.z))
        end
    end
    log(string.format("[ChargerDebug] passable_count=%d, passable_anchors=%s", #passable_charger_anchors, format_anchor_list(passable_charger_anchors)))

    -- 2. 随机打乱后取前 desired_charger 个
    shuffle_inplace(passable_charger_anchors)
    log(string.format("[ChargerDebug] shuffled_passable_anchors=%s", format_anchor_list(passable_charger_anchors)))

    local picked_charger_anchors = {}
    for i = 1, math.min(desired_charger, #passable_charger_anchors) do
        table.insert(picked_charger_anchors, passable_charger_anchors[i])
    end
    log(string.format("[ChargerDebug] picked_count=%d, picked_anchors=%s", #picked_charger_anchors, format_anchor_list(picked_charger_anchors)))

    -- 3. 首轮仍移除未被选中的预设充电桩源对象，后续同图 reset 直接复用缓存重新随机
    if not preset_from_cache and ChargerElement.prune_drone_preset_chargers then
        local removed = ChargerElement.prune_drone_preset_chargers(ECS, picked_charger_anchors)
        if removed > 0 then
            log(string.format("[Charger] Pruned %d unselected preset charger anchor objects from map", removed))
        end
    end

    -- 目标数量严格等于配置值；不足时再用 Voronoi 补齐
    local target_charger = desired_charger

    -- 2. 锚点不足时，用 Voronoi 补齐
    local charger_gap = target_charger - #picked_charger_anchors
    if charger_gap > 0 then
        log(string.format("[Charger] Preset anchors=%d, need %d more, using Voronoi", #picked_charger_anchors, charger_gap))

        -- 收集候选左上角锚点（复用驿站的候选池，但排除已选驿站附近）
        local anchors = collect_anchor_candidates()
        local anchor_candidates = {}

        -- 道路锚点优先
        for _, anchor in ipairs(anchors.road_anchors) do
            table.insert(anchor_candidates, anchor)
        end
        if #anchor_candidates < charger_gap * 10 then
            for _, anchor in ipairs(anchors.other_anchors) do
                table.insert(anchor_candidates, anchor)
            end
        end

        if #anchor_candidates == 0 then
            log_error("[Charger] No passable anchors for Voronoi generation")
        else
            -- 构建已有锚点集（预设充电桩 + 所有驿站位置）
            local existing_anchors = {}
            for _, p in ipairs(picked_charger_anchors) do
                table.insert(existing_anchors, { x = p.x, z = p.z })
            end
            for _, s in ipairs(game_state.stations) do
                table.insert(existing_anchors, { x = s.x, z = s.z })
            end

            -- K-means++ 风格选取锚点
            local voronoi_anchors = select_spread_anchors_kmeans(anchor_candidates, charger_gap, STATION_MIN_DISTANCE, existing_anchors)

            log(string.format("[Charger] Voronoi generated %d anchors", #voronoi_anchors))

            for _, anchor in ipairs(voronoi_anchors) do
                table.insert(picked_charger_anchors, {
                    x = anchor.x,
                    z = anchor.z,
                    w = STATION_SIZE,
                    h = STATION_SIZE,
                    range = 1.5,
                })
            end

        end
    end

    log(string.format("[Charger] Total chargers: %d (desired: %d, target: %d)", #picked_charger_anchors, desired_charger, target_charger))

    for _, p in ipairs(picked_charger_anchors) do
        local cw = math.max(1, math.floor((tonumber(p.w) or STATION_SIZE) + 1e-6))
        local chh = math.max(1, math.floor((tonumber(p.h) or STATION_SIZE) + 1e-6))
        local anchor_x = p.x
        local anchor_z = p.z
        local spawn_center_x = anchor_x + math.floor(cw / 2)
        local spawn_center_z = anchor_z + math.floor(chh / 2)
        local eid, composed_cells = GridUnit.spawn_tiled_units({
            center_x = spawn_center_x,
            center_z = spawn_center_z,
            size = STATION_SIZE,
            unit_name = CHARGER_UNIT_OBJECT_NAME,
            tags = { "charger", CHARGER_UNIT_OBJECT_NAME },
            color = { 255, 215, 0, 255 },
            center_label = "充电桩",
            in_bounds_fn = is_in_bounds,
            key_fn = key_xy,
        })

        -- 3x3 充电桩逻辑 = 多个 1x1 单元（直接复用 GridUnit 返回的 cell_map）
        for k, _ in pairs(composed_cells or {}) do
            game_state.charger_cells[k] = true
        end

        -- 默认充电范围与占地一致：3x3 半径 1.5 格
        table.insert(game_state.chargers, {
            x = anchor_x,
            z = anchor_z,
            w = cw,
            h = chh,
            range = p.range or 1.5,
            entity_id = eid,
        })

    end
end

local function get_station_id_at_cell(gx, gz)
    -- 优先按 1x1 组合单元判定（3x3 = 多个 1x1）
    local sid = game_state.station_cells and game_state.station_cells[key_xy(gx, gz)]
    if sid ~= nil then
        return sid
    end

    -- 兼容回退：按左上角锚点+占地尺寸判定
    for _, s in ipairs(game_state.stations) do
        local sw = math.max(1, math.floor((tonumber(s.w) or STATION_SIZE) + 1e-6))
        local sh = math.max(1, math.floor((tonumber(s.h) or STATION_SIZE) + 1e-6))
        if gx >= s.x and gx < (s.x + sw) and gz >= s.z and gz < (s.z + sh) then
            return s.id
        end
    end
    return nil
end


local function refill_battery_full()
    if game_state.battery >= game_state.battery_max then
        return false
    end

    game_state.battery = game_state.battery_max
    if game_state.player_id then
        entity.set_property(game_state.player_id, "battery", game_state.battery)
        entity.set_property(game_state.player_id, "battery_max", game_state.battery_max)
    end

    return true
end


local function refill_packages()
    -- 规则：仓库包裹无限；无人机容量固定为 3
    -- 规则：拿取时包裹编号随机，尽量不重复；当可用驿站数不足以填满时允许重复
    local cap = 3

    if not game_state.packages then
        game_state.packages = {}
    end

    if not game_state.stations or #game_state.stations == 0 then
        game_state.packages = {}
        sync_drone_visual()
        return
    end

    local before = #game_state.packages
    while #game_state.packages < cap do
        local sid = pick_station_id_for_package()
        if sid == nil then
            break
        end
        table.insert(game_state.packages, sid)
    end

    sync_drone_visual()

    local added = #game_state.packages - before
    local package_list = (#game_state.packages > 0) and table.concat(game_state.packages, ",") or "(空)"
    if added > 0 then
        log(string.format("[Warehouse] Refilled packages: +%d, now=%d (%s)", added, #game_state.packages, package_list))
        if ui and ui.toast then
            ui.toast("仓库补给(+" .. tostring(added) .. ") 当前包裹: " .. package_list, 2.0)
        end
    end
end





local function try_deliver_at_station(station_id)
    if not station_id then
        return 0
    end

    if not game_state.packages or #game_state.packages == 0 then
        return 0
    end

    local remaining = {}
    local delivered = 0

    for _, sid in ipairs(game_state.packages) do
        if sid == station_id then
            delivered = delivered + 1
        else
            table.insert(remaining, sid)
        end
    end

    if delivered > 0 then
        game_state.packages = remaining
        sync_drone_visual()
        game_state.delivered = game_state.delivered + delivered
        game_state.score = game_state.score + delivered * (usr_config.score_per_delivery or 100)
        log(string.format("[Delivery] Station #%d delivered=%d, score=%d", station_id, delivered, game_state.score))
        if ui and ui.toast then
            ui.toast(string.format("投递成功: 驿站#%d (+%d)", station_id, delivered * (usr_config.score_per_delivery or 100)),
                1.5)
        end
    end



    return delivered
end

local function is_in_charger_range_xy(px, pz)
    -- 优先按 1x1 组合单元判定（N×N = 多个 1x1）
    local gx = math.floor((px or 0) + 1e-6)
    local gz = math.floor((pz or 0) + 1e-6)
    if game_state.charger_cells and game_state.charger_cells[key_xy(gx, gz)] then
        return true
    end

    -- 回退判定：按充电桩中心 + range 做距离检测（避免 cell_map 与位置采样时序不一致）
    for _, c in ipairs(game_state.chargers or {}) do
        local cw = math.max(1, math.floor((tonumber(c.w) or STATION_SIZE) + 1e-6))
        local chh = math.max(1, math.floor((tonumber(c.h) or STATION_SIZE) + 1e-6))
        local cx = (tonumber(c.x) or 0) + math.floor(cw / 2) + 0.5
        local cz = (tonumber(c.z) or 0) + math.floor(chh / 2) + 0.5
        local r = tonumber(c.range) or 1.5
        local dx = (px or 0) - cx
        local dz = (pz or 0) - cz
        if (dx * dx + dz * dz) <= (r * r + 1e-6) then
            return true
        end
    end

    return false
end

local function apply_charger_effect(px, pz)
    local p = entity.get_player()
    if (not p) and (px == nil or pz == nil) then
        game_state.in_charger = false
        return false
    end

    local check_x = (px ~= nil) and px or p.x
    local check_z = (pz ~= nil) and pz or p.z
    local now_in_charger = is_in_charger_range_xy(check_x, check_z)
    local entered = now_in_charger and (not game_state.in_charger)

    if entered then
        game_state.charger_charge_count = (game_state.charger_charge_count or 0) + 1
    end

    if now_in_charger then
        local charged = refill_battery_full()
        -- 每步都会自动充满；提示仅在"进入范围"那一步弹一次
        if entered and ui and ui.toast then
            if charged then
                ui.toast("进入充电桩范围：电量已充满", 1.2)
            else
                ui.toast("进入充电桩范围：电量已满", 1.2)
            end
        end
    end

    game_state.in_charger = now_in_charger
    return now_in_charger
end



local function find_any_passable_position()
    -- 从中心向外找一个可通行点
    local cx = game_state.map_width // 2
    local cz = game_state.map_height // 2

    for radius = 0, 80 do
        local found = nil
        GridUnit.for_each_square_radius(cx, cz, radius, function(x, z)
            if not found and is_in_bounds(x, z) and is_passable_for_drone(x, z) then
                found = { x = x, z = z }
            end
        end)
        if found then
            return found
        end
    end

    return nil
end

-- ============================================================================
-- 实体生成
-- ============================================================================

local function spawn_drone()
    local spawn_points = DronePlayerElement.query_spawn_points(ECS)

    local spawn_pos = { x = map.get_width() / 2, z = map.get_height() / 2 }
    if #spawn_points > 0 then
        if usr_config.start_random then
            shuffle_inplace(spawn_points)
            for _, p in ipairs(spawn_points) do
                if is_passable_for_drone(p.x, p.z) then
                    spawn_pos = p
                    break
                end
            end
        else
            local idx = clamp(usr_config.start or 1, 1, #spawn_points)
            local p = spawn_points[idx]
            if is_passable_for_drone(p.x, p.z) then
                spawn_pos = p
            else
                -- 固定点不可达时，尝试在预设点中找第一个可达点
                for _, cand in ipairs(spawn_points) do
                    if is_passable_for_drone(cand.x, cand.z) then
                        spawn_pos = cand
                        break
                    end
                end
            end
        end
    else
        -- 没有预设出生点时，优先回退任意可通行格
        local fallback = find_any_passable_position()
        if fallback then
            spawn_pos = fallback
        end
    end

    if not spawn_pos or not is_passable_for_drone(spawn_pos.x, spawn_pos.z) then
        log_warn("[Spawn] No valid drone spawn points, falling back to any passable cell")
        spawn_pos = find_any_passable_position()
    end

    if not spawn_pos then
        log_error("[Spawn] Failed to pick a drone spawn")
        game_state.task_status = "failed"
        game_state.fail_reason = "无人机生成失败"
        return false
    end

    local profile = RobotProfiles.get("drone_delivery", "player") or {}

    local player_id = entity.spawn(
        profile.tag or "drone_player",
        spawn_pos.x + 0.5,
        spawn_pos.z + 0.5,
        merge_tags({ "player", "agent", "drone" }, profile.tags),
        {
            collision = {
                enabled = true,
                shape = {
                    type = "circle",
                    radius = tonumber(profile.collision_size and profile.collision_size.radius) or 0.4,
                },
            },
            visual = {
                display_name = profile.name or "无人机",
                color = profile.color or { 0, 200, 255, 255 },
                visible = true,
            },
        }
    )

    if not player_id then
        log_error("[Spawn] Failed to create drone entity")
        game_state.task_status = "failed"
        game_state.fail_reason = "无人机实体创建失败"
        return false
    end

    DronePlayerElement.consume_source(spawn_pos)

    game_state.player_id = player_id
    game_state.player = entity.get_player()
    game_state.player_fly_over_bonus = tonumber(profile.fly_over_bonus) or 0.0
    game_state.player_battery_cost_per_step = clamp(math.floor((tonumber(profile.battery_cost_per_step) or (usr_config.battery_per_step or 1)) + 1e-6), 0, 100)

    -- 初始化实体属性（用于 UI/调试）
    game_state.battery = game_state.battery_max
    entity.set_property(game_state.player_id, "battery", game_state.battery)
    entity.set_property(game_state.player_id, "battery_max", game_state.battery_max)
    entity.set_property(game_state.player_id, "score", game_state.score)
    entity.set_property(game_state.player_id, "robot_profile", profile.class_name or "standard")
    entity.set_property(game_state.player_id, "fly_over_bonus", game_state.player_fly_over_bonus)
    entity.set_property(game_state.player_id, "battery_cost_per_step", game_state.player_battery_cost_per_step)

    return game_state.player ~= nil
end

-- ============================================================================
-- 核心逻辑：移动与任务状态
-- ============================================================================

local function finish_step_logic(target_gx, target_gy)
    game_state.current_step = game_state.current_step + 1
    game_state.last_step_reward = 0

    -- 电量消耗
    game_state.battery = game_state.battery - (game_state.player_battery_cost_per_step or usr_config.battery_per_step or 1)
    if game_state.battery <= 0 then
        game_state.battery = 0
        game_state.task_status = "failed"
        game_state.fail_reason = "电量耗尽"
    end

    if game_state.player_id then
        entity.set_property(game_state.player_id, "battery", game_state.battery)
    end

    -- 充电桩：范围内每走一步都会自动充满（提示仅进入范围那一步弹一次）
    apply_charger_effect(target_gx + 0.5, target_gy + 0.5)

    -- 仓库：区域内每走一步都补足包裹额度 + 充满能量
    -- 提示策略：只在"进入仓库"的那一步提示一次（即使能量原本已满，也提示，方便确认机制生效）
    local now_in_warehouse = (game_state.warehouse_cells[key_xy(target_gx, target_gy)] == true)
    local entered_warehouse = now_in_warehouse and (not game_state.in_warehouse)

    if now_in_warehouse then
        refill_packages()
        local charged = refill_battery_full()

        if entered_warehouse then
            game_state.warehouse_return_count = (game_state.warehouse_return_count or 1) + 1
            game_state.charger_charge_count = (game_state.charger_charge_count or 0) + 1
        end

        if entered_warehouse and ui and ui.toast then
            local pkg_list = (#(game_state.packages or {}) > 0)
                and table.concat(game_state.packages, ",")
                or "(空)"
            local station_cnt = (game_state.stations and #game_state.stations) or 0
            local battery_hint = charged and "电量已充满" or "电量已满"

            if station_cnt == 0 then
                ui.toast("进入仓库：" .. battery_hint .. "；未配置驿站点位，无法生成包裹", 2.0)
            else
                ui.toast(
                    "进入仓库：" .. battery_hint .. "；包裹 " .. tostring(#(game_state.packages or {})) .. "/3 " .. pkg_list,
                    2.0
                )
            end
        end
    end

    game_state.in_warehouse = now_in_warehouse




    -- 驿站投递：进入驿站 3x3 占地范围则触发（提示仅进入那一步弹一次）
    local sid = get_station_id_at_cell(target_gx, target_gy)
    if sid == nil then
        game_state.last_station_entered = nil
    else
        local entered_station = (game_state.last_station_entered ~= sid)
        if entered_station then
            local delivered = try_deliver_at_station(sid)
            if delivered > 0 then
                game_state.last_step_reward = 0
            elseif ui and ui.toast then
                ui.toast("进入驿站#" .. tostring(sid) .. "：无可投递包裹", 1.2)
            end
        end
        game_state.last_station_entered = sid
    end


    -- 官方无人机：与玩家同频，玩家每步动作后移动一步
    update_official_drones(nil)


    -- 道具点位已移除：不再通过点位拾取包裹

    -- 碰撞：任何 enemy 在 3*3 范围内即失败（包括官方无人机/机器人）
    check_collision_player_vs_enemies()



    if game_state.player_id then
        entity.set_property(game_state.player_id, "score", game_state.score)
    end

    -- 步数终止
    if game_state.current_step >= game_state.max_step then
        if game_state.task_status == "running" then
            game_state.task_status = "completed"
        end
    end

    return true
end


local function move_player(direction)
    local p = entity.get_player()
    if not p then return false end

    local dv = DIRECTION_VECTORS[direction]
    if not dv then
        log_warn(string.format("[Move] Invalid direction: %s", tostring(direction)))
        return false
    end

    local gx = math.floor(p.x)
    local gz = math.floor(p.z)

    local tx = gx + dv.x
    local tz = gz + dv.z

    local final_tx = gx
    local final_tz = gz

    if can_move_diagonally(gx, gz, tx, tz) then
        final_tx = tx
        final_tz = tz
    end

    local moved = not (final_tx == gx and final_tz == gz)

    if moved then
        entity.move_to(p.id, final_tx + 0.5, final_tz + 0.5)
    end

    -- 一次动作对应一步（即使撞墙不动，也消耗步数/能量，避免"卡墙刷无代价"）
    return finish_step_logic(final_tx, final_tz)
end

local function teleport_player_to_grid(target_x, target_z)
    if game_state.task_status ~= "running" then
        return {
            ok = false,
            message = "当前任务未运行，无法传送",
        }
    end

    local player = entity.get_player()
    if not player or not game_state.player_id then
        return {
            ok = false,
            message = "未找到 player，无法传送",
        }
    end

    local gx = tonumber(target_x)
    local gz = tonumber(target_z)
    if gx == nil or gz == nil then
        return {
            ok = false,
            message = "坐标必须为数字",
        }
    end

    gx = math.floor(gx)
    gz = math.floor(gz)

    if not is_in_bounds(gx, gz) then
        return {
            ok = false,
            message = string.format("目标坐标越界: (%d, %d)", gx, gz),
        }
    end

    if not is_passable_for_drone(gx, gz) then
        return {
            ok = false,
            message = string.format("目标坐标不可通行: (%d, %d)", gx, gz),
        }
    end

    entity.set_position(game_state.player_id, gx + 0.5, gz + 0.5)
    game_state.player = entity.get_player()

    local now_in_warehouse = (game_state.warehouse_cells[key_xy(gx, gz)] == true)
    local entered_warehouse = now_in_warehouse and (not game_state.in_warehouse)
    if now_in_warehouse then
        refill_packages()
        refill_battery_full()
        if entered_warehouse then
            game_state.warehouse_return_count = (game_state.warehouse_return_count or 1) + 1
            game_state.charger_charge_count = (game_state.charger_charge_count or 0) + 1
        end
    end
    game_state.in_warehouse = now_in_warehouse

    apply_charger_effect(gx + 0.5, gz + 0.5)

    local sid = get_station_id_at_cell(gx, gz)
    if sid == nil then
        game_state.last_station_entered = nil
    else
        local entered_station = (game_state.last_station_entered ~= sid)
        if entered_station then
            try_deliver_at_station(sid)
        end
        game_state.last_station_entered = sid
    end

    check_collision_player_vs_enemies()

    if game_state.player_id then
        entity.set_property(game_state.player_id, "score", game_state.score)
        entity.set_property(game_state.player_id, "battery", game_state.battery)
        entity.set_property(game_state.player_id, "battery_max", game_state.battery_max)
    end

    if game_state.map_viz_visible then
        game_state.last_map_viz_cx = gx
        game_state.last_map_viz_cz = gz
        game_state.last_map_viz_vr = game_state.last_map_viz_vr or VISION_RANGE
        game_state.last_map_viz_grid = nil
    end

    return {
        ok = true,
        message = string.format("已传送到 (%d, %d)", gx, gz),
    }
end


-- ============================================================================
-- 核心接口（供 Python / 调试使用）
-- ============================================================================

-- 视野范围常量
local VISION_RANGE = 10  -- 以玩家为中心，21x21 格子

local _frame_builder = nil

local function ensure_frame_builder()
    if _frame_builder then
        return _frame_builder
    end

    _frame_builder = DroneDeliveryFrame.new({
        entity = entity,
        get_game_state = function() return game_state end,
        get_usr_config = function() return usr_config end,
        deep_copy = deep_copy,
        is_in_bounds = is_in_bounds,
        is_passable_for_drone = is_passable_for_drone,
        station_size = STATION_SIZE,
        vision_range = VISION_RANGE,
    })

    return _frame_builder
end

--- collect_frame_state - 返回一个稳定的帧数据（供 map_runtime_py.step / 调试使用）
--- 说明：编辑器模式下 on_update 的返回值会被忽略；Python headless 模式会读取该返回值。
function collect_frame_state()
    local task = get_task_state()

    local truncated = (task.current_step >= task.max_step)
    local done = (task.status ~= "running") and (not truncated)

    local frame_no = task.current_step or 0
    local map_id = (game_state and game_state.current_map_id) or (usr_config and usr_config.map_id) or 0
    local env_id = tostring(map_id)
    local frame_builder = ensure_frame_builder()
    
    -- 构建新增的结构
    local frame_state = frame_builder.build_frame_state()
    local env_info = frame_builder.build_env_info()
    local map_info = frame_builder.build_map_info()

    return {
        reward = {
            env_id = env_id,
            frame_no = frame_no,
            reward = 0,
        },

        obs = {
            env_id = env_id,
            frame_no = frame_no,
            observation = {
                step_no = frame_no,
                frame_state = frame_state,
                env_info = env_info,
                map_info = map_info,
                legal_act = { 1, 1, 1, 1, 1, 1, 1, 1 },
            },
            extra_info = {
                frame_state = frame_state,
                map_id = map_id,
                result_code = 0,
                result_message = "",
            },
            terminated = done,
            truncated = truncated,
        },
      
    }
end

local function init_environment()
    rebuild_object_height_cache()
    init_warehouse_cells()
    init_stations_and_chargers()

    -- 道具点位已移除：开局包裹数量为 0（需要进入仓库后才会补给）
    game_state.packages = {}


    -- 官方无人机：按出生点生成并开始巡逻
    spawn_official_drones()

    sync_drone_visual()
end




function reset(user_config)
    usr_config = merge_config(DEFAULT_CONFIG, user_config or {})

    -- 初始化随机种子（必须在所有随机操作之前）
    -- 约定：random_seed>0 时使用外部传入值；random_seed==-1 时自动生成随机种子；其他情况保持当前随机状态不变
    local raw_seed = usr_config.random_seed
    local seed = tonumber(raw_seed)
    local parsed_seed = seed
    if seed ~= nil and seed > 0 then
        seed = math.floor(seed + 1e-6)
        math.randomseed(seed)
    elseif seed == -1 then
        local time_seed = os.time()
        local clock_seed = math.floor((os.clock() % 1) * 1000000)
        seed = time_seed + clock_seed
        math.randomseed(seed)
    else
        seed = nil
    end
    log(string.format("[Reset] Random seed input=%s, parsed=%s, applied=%s", tostring(raw_seed), tostring(parsed_seed), tostring(seed)))

    -- 地图选择（注意：通常由 Python 端处理 map_random 并加载对应地图；这里仅保留配置与日志）
    if usr_config.map_random == nil then
        usr_config.map_random = true
    end
    usr_config.map_id = clamp(math.floor(parse_number(usr_config.map_id, 1) + 1e-6), 1, 999999)
    if usr_config.map_random then
        log(string.format("[Config] Map selected by Python: map_id=%d", usr_config.map_id))
    else
        log(string.format("[Config] Fixed map_id: %d", usr_config.map_id))
    end

    usr_config.max_step = clamp(usr_config.max_step or 1000, 1, 2000)
    usr_config.fly_over_height = parse_number(usr_config.fly_over_height, 2.0)
    usr_config.station_count = clamp(usr_config.station_count or 4, 1, 10)
    usr_config.charger_count = clamp(usr_config.charger_count or 2, 1, 4)
    -- 规则：无人机每次最多携带 3 个包裹（固定，不从配置读取）
    usr_config.package_capacity = 3

    usr_config.score_per_delivery = clamp(usr_config.score_per_delivery or 100, 1, 100000)
    usr_config.battery_max = clamp(usr_config.battery_max or 100, 1, 1000)
    usr_config.battery_per_step = clamp(usr_config.battery_per_step or 1, 0, 100)

    log(string.format(
        "[Config] Effective knobs: station_count=%d, charger_count=%d, drone_count=%d, max_step=%d, map_id=%d",
        usr_config.station_count,
        usr_config.charger_count,
        clamp(usr_config.drone_count or 4, 0, 10),
        usr_config.max_step,
        usr_config.map_id
    ))

    -- 可观测性：打印“脚本归一化后的配置键”
    do
        local keys = collect_sorted_keys(usr_config)
        log(string.format("[Config] Normalized keys(%d): %s", #keys, table.concat(keys, ", ")))
    end
    key_diff_report(__drone_injected_keys, usr_config, "drone_delivery")

    _step_action = nil
    _step_extra_info = nil
    _passability_api = nil

    entity.clear()

    game_state.player = nil
    game_state.player_id = nil
    game_state.map_width = map.get_width()
    game_state.map_height = map.get_height()
    game_state.current_map_id = usr_config.map_id

    game_state.current_step = 0
    game_state.max_step = usr_config.max_step
    game_state.task_status = "running"
    game_state.fail_reason = ""

    game_state.score = 0
    game_state.delivered = 0
    game_state.last_step_reward = 0
    game_state.battery_max = usr_config.battery_max
    game_state.battery = usr_config.battery_max
    game_state.player_fly_over_bonus = 0.0
    game_state.player_battery_cost_per_step = usr_config.battery_per_step
    game_state.packages = {}
    game_state.in_warehouse = false
    game_state.in_charger = false
    game_state.last_station_entered = nil
    game_state._warned_no_set_color = nil
    game_state.warehouse_return_count = 1 -- 返回仓库次数（起始算一次）
    game_state.charger_charge_count = 0 -- 充电次数（进入充电桩范围计一次）

    -- 调试面板：每次 reset 都清空，避免停止后再次播放出现状态残留
    game_state.map_viz_visible = false
    game_state.last_map_viz_grid = nil
    game_state.last_map_viz_cx = nil
    game_state.last_map_viz_cz = nil
    game_state.last_map_viz_vr = nil


    init_environment()

    if not spawn_drone() then
        return collect_frame_state()
    end

    -- 检查玩家是否出生在仓库区域：如果是，自动补充包裹和能量
    do
        local p = entity.get_player()
        if p then
            local gx = math.floor(p.x)
            local gz = math.floor(p.z)
            local spawn_in_warehouse = (game_state.warehouse_cells[key_xy(gx, gz)] == true)
            
            if spawn_in_warehouse then
                -- 玩家出生在仓库，自动补充包裹
                game_state.in_warehouse = true
                refill_packages()
                refill_battery_full()
                
                local pkg_list = (#(game_state.packages or {}) > 0)
                    and table.concat(game_state.packages, ",")
                    or "(空)"
                log(string.format("[Init] Spawned in warehouse, auto-refilled packages: %s", pkg_list))
                if ui and ui.toast then
                    ui.toast("出生在仓库：已自动补充包裹 " .. pkg_list, 2.0)
                end
            else
                game_state.in_warehouse = false
            end
            
            game_state.in_charger = is_in_charger_range_xy(p.x, p.z)
            game_state.last_station_entered = get_station_id_at_cell(gx, gz)
        end
    end

    sync_drone_visual()


    log(string.format(
        "[Init] map=%dx%d, stations=%d, chargers=%d, fly_over_height=%.2f",
        game_state.map_width,
        game_state.map_height,
        #game_state.stations,
        #game_state.chargers,
        usr_config.fly_over_height
    ))

    return collect_frame_state()
end

function execute_agent_action(action)
    if game_state.task_status ~= "running" then
        return false
    end

    if action == nil then
        return false
    end

    -- 只支持 0-7（8方向移动）
    if action < 0 or action > 7 then
        return false
    end

    return move_player(action)
end

local function build_status_api()
    return DroneDeliveryStatus.new({
        get_game_state = function() return game_state end,
        get_usr_config = function() return usr_config end,
        deep_copy = deep_copy,
        get_cell_info_text = function()
            local p = entity.get_player()
            if not p then
                return "格子ID: nil  位置: (nil)  类型: (nil)"
            end

            local px = math.floor(p.x)
            local pz = math.floor(p.z)
            if not is_in_bounds(px, pz) then
                return string.format("格子ID: nil  位置: [%d,%d]  类型: 越界", px, pz)
            end

            local cell_id = pz * game_state.map_width + px
            local type_id = (map.get_cell_type and map.get_cell_type(px, pz)) or 0
            local type_name = ({
                [0] = "可通行",
                [1] = "不可通行",
                [2] = "污渍",
                [3] = "道路",
                [4] = "桥",
                [5] = "草地",
            })[type_id] or "未知"
            local walkable = is_base_walkable(px, pz)
            local pass_api = ensure_passability_api()
            local h_cell = pass_api.get_cell_height(px, pz)
            local h_obj = pass_api.get_obj_height_at(px, pz)
            local h_max = pass_api.get_max_height_at(px, pz)

            return string.format(
                "格子ID: %d  位置: [%d,%d]  类型: %s(type_id=%s)  可通行=%s  高度=%.1f(地%.1f/物%.1f)",
                cell_id,
                px,
                pz,
                type_name,
                tostring(type_id),
                walkable and "是" or "否",
                h_max,
                h_cell,
                h_obj
            )
        end,
    })
end

function get_task_state()
    return build_status_api().get_task_state()
end

function get_hud_text()
    return build_status_api().get_hud_text()
end

function get_agent_observation()
    local p = entity.get_player()
    if not p then
        return nil
    end

    local pgx = math.floor(p.x)
    local pgz = math.floor(p.z)

    -- 观测：位置/能量/包裹 + 局部 21x21 可通行 + 局部高度
    local vision = {}
    local heights = {}

    for dy = -10, 10 do
        local row = {}
        local row_h = {}
        for dx = -10, 10 do
            local x = pgx + dx
            local z = pgz + dy

            local v = 0
            local h = 0.0
            if is_in_bounds(x, z) then
                v = is_passable_for_drone(x, z) and 1 or 0
                h = ensure_passability_api().get_max_height_at(x, z)
            end

            table.insert(row, v)
            table.insert(row_h, h)
        end
        table.insert(vision, row)
        table.insert(heights, row_h)
    end

    local stations = {}
    for _, s in ipairs(game_state.stations) do
        table.insert(stations, { id = s.id, x = s.x, z = s.z })
    end

    return {
        player = {
            x = pgx,
            z = pgz,
            battery = game_state.battery,
            battery_max = game_state.battery_max,
            packages = deep_copy(game_state.packages),
        },
        stations = stations,
        vision_grid = vision,
        height_grid = heights,
        task = {
            current_step = game_state.current_step,
            max_step = game_state.max_step,
            score = game_state.score,
            delivered = game_state.delivered,
        },
    }
end

-- ============================================================================
-- 运行时回调
-- ============================================================================

local _world_api = nil

local function ensure_world_api()
    if _world_api then
        return _world_api
    end

    _world_api = DroneDeliveryWorld.new({
        log = log,
        log_warn = log_warn,
        control = Control,
        entity = entity,
        reset = reset,
        execute_agent_action = execute_agent_action,
        collect_frame_state = collect_frame_state,
        update_official_drones = update_official_drones,
        check_collision_player_vs_enemies = check_collision_player_vs_enemies,
        spawn_drone = spawn_drone,
        rebuild_object_height_cache = rebuild_object_height_cache,
        get_game_state = function() return game_state end,
        get_step_action = function() return _step_action end,
        clear_step_action = function()
            _step_action = nil
            _step_extra_info = nil
        end,
        frame_builder = {
            visualize_map_info_grid = function(d)
                local p = entity and entity.get_player and entity.get_player()
                if not p then return false end
                local cx = math.floor(p.x)
                local cz = math.floor(p.z)
                local vr = (d and d.vision_range) or VISION_RANGE
                local frame_builder = ensure_frame_builder()
                local grid_data = frame_builder.build_map_info()
                game_state.last_map_viz_grid = grid_data
                game_state.last_map_viz_cx = cx
                game_state.last_map_viz_cz = cz
                game_state.last_map_viz_vr = vr
                return true
            end,
            clear_map_info_visualization = function()
                game_state.last_map_viz_grid = nil
                game_state.last_map_viz_cx = nil
                game_state.last_map_viz_cz = nil
                game_state.last_map_viz_vr = nil
            end,
        },
    })

    return _world_api
end

function on_start()
    return ensure_world_api().on_start()
end

function on_update(dt)
    return ensure_world_api().on_update(dt)
end

-- 运行中编辑（编辑器增量放置/删除物体）可选回调
-- 约定：回调可缺失；若实现，内部应容错，不影响运行
function on_runtime_object_upserted(obj)
    return ensure_world_api().on_runtime_object_upserted(obj)
end

function on_runtime_object_removed(instance_id)
    return ensure_world_api().on_runtime_object_removed(instance_id)
end

function on_stop()
    return ensure_world_api().on_stop()
end

-- Rust 调试面板读取入口（全局函数）
function get_map_viz_grid()
    return {
        grid = game_state.last_map_viz_grid,
        cx = game_state.last_map_viz_cx,
        cz = game_state.last_map_viz_cz,
        vr = game_state.last_map_viz_vr,
    }
end

function teleport_player_to(x, z)
    return teleport_player_to_grid(x, z)
end

log("drone script loaded successfully")
