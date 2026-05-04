local M = {}
local RobotProfiles = require("ecs.robot_profiles")
local EnemyProximity = require("ecs.elements.enemy_proximity")
local ECSCommon = require("ecs.common")

M.key = "enemy_spawn"
local merge_tags = ECSCommon.merge_tags
local has_component = ECSCommon.has_component

function M.match(obj)
    local name = tostring(obj and obj.name or "")
    if name == "spawn_point_npc"
        or name == "drone_delivery_official_spawn"
        or name == "robot_vacuum_npc"
        or name == "drone_delivery_official"
    then
        return true
    end

    local props = (obj and obj.properties) or {}
    local spawn_type = tostring(props.spawn_type or "")
    if spawn_type == "npc" or spawn_type == "drone_delivery_official" then
        return true
    end

    return has_component(obj, "spawn.npc")
        or has_component(obj, "spawn.drone_official")
end

function M.query_robot_npc_spawn_points(ECS)
    local points = ECS.query_points({
        component = "spawn.npc",
        names = { "npc_spawn", "spawn_point_npc", "robot_vacuum_npc" },
        categories = { "Spawn", "spawn", "Interactive", "interactive" },
        predicate = function(obj)
            return obj.name == "npc_spawn" or obj.name == "spawn_point_npc" or obj.name == "robot_vacuum_npc"
                or (obj.properties and (obj.properties.spawn_type == "npc" or obj.properties.npc == "true"))
        end,
    })

    local spawn_points = {}
    for _, p in ipairs(points) do
        table.insert(spawn_points, { x = p.x, z = p.z, obj = p.obj })
    end

    return spawn_points
end

function M.query_drone_official_spawn_points(ECS)
    local name_candidates = { "drone_delivery_official_spawn", "drone_delivery_official" }
    local spawn_type_candidates = { "drone_delivery_official" }

    local point_rows = ECS.query_points({
        component = "spawn.drone_official",
        names = name_candidates,
        categories = { "Spawn", "spawn", "Interactive", "interactive" },
        predicate = function(obj)
            for _, n in ipairs(name_candidates) do
                if obj.name == n then
                    return true
                end
            end
            if obj.properties and obj.properties.spawn_type then
                for _, st in ipairs(spawn_type_candidates) do
                    if obj.properties.spawn_type == st then
                        return true
                    end
                end
            end
            return false
        end,
    })

    local points = {}
    for _, p in ipairs(point_rows) do
        table.insert(points, { x = p.x, z = p.z, properties = p.obj and p.obj.properties or nil, obj = p.obj })
    end

    return points
end

function M.consume_source_instance(point, instance_name)
    local obj = point and point.obj
    if obj and obj.name == instance_name and map.remove_object and obj.instance_id ~= nil then
        map.remove_object(obj.instance_id)
    end
end

function M.has_enemy_near_grid(player_gx, player_gy, range)
    return EnemyProximity.has_enemy_near_grid(entity, player_gx, player_gy, range)
end

function M.has_enemy_near_player_grid(range)
    return EnemyProximity.has_enemy_near_player_grid(entity, range)
end

function M.fail_if_enemy_near_grid(game_state, player_gx, player_gy, options)
    return EnemyProximity.fail_if_enemy_near_grid(entity, ui, game_state, player_gx, player_gy, options)
end

function M.fail_if_enemy_near_player(game_state, options)
    return EnemyProximity.fail_if_enemy_near_player(entity, ui, game_state, options)
end

function M.build(ctx, obj, index)
    local x, z = ctx.helpers.center_from_obj(obj)
    local name = tostring(obj and obj.name or "")
    local props = (obj and obj.properties) or {}
    local spawn_type = tostring(props.spawn_type or "")

    local scene = "robot_vacuum"
    local role = "npc"
    if name == "spawn_point_npc" or name == "robot_vacuum_npc" or spawn_type == "npc" or has_component(obj, "spawn.npc") then
        scene = "robot_vacuum"
        role = "npc"
    elseif name == "drone_delivery_official_spawn" or name == "drone_delivery_official" or spawn_type == "drone_delivery_official" or has_component(obj, "spawn.drone_official") then
        scene = "drone_delivery"
        role = "official"
    end

    local profile = RobotProfiles.get(scene, role) or {}
    local collision_radius = tonumber(profile.collision_size and profile.collision_size.radius) or 0.35
    local spawn_name = (profile.tag or "enemy") .. "_" .. tostring(index)
    local display_name = profile.name or "敌人"
    local color = profile.color or {255, 60, 60, 255}

    local id = entity.spawn(
        spawn_name,
        x,
        z,
        merge_tags({"enemy", "builder_enemy"}, profile.tags),
        {
            collision = {
                enabled = true,
                shape = { type = "circle", radius = collision_radius },
            },
            visual = {
                display_name = display_name,
                color = color,
                visible = true,
            },
        }
    )

    if id then
        table.insert(ctx.state.enemy_ids, id)
        entity.set_property(id, "robot_profile", profile.class_name or "standard")
        if obj and (obj.name == "robot_vacuum_npc" or obj.name == "drone_delivery_official") and map.remove_object and obj.instance_id ~= nil then
            map.remove_object(obj.instance_id)
        end
        log(string.format("[world_builder] enemy spawned id=%s from %s at (%.1f, %.1f)", tostring(id), tostring(obj and obj.name), x, z))
    end

    return { key = M.key, index = index, entity_id = id }
end

return M
