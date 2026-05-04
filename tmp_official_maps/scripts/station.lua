local M = {}

local function obj_top_left(obj)
    local x = tonumber(obj and obj.x)
    local z = tonumber(obj and obj.z)
    if (x == nil or z == nil) and type(obj and obj.position) == "table" then
        x = x or tonumber(obj.position[1]) or tonumber(obj.position.x)
        z = z or tonumber(obj.position[2]) or tonumber(obj.position.z)
    end
    return math.floor((x or 0) + 1e-6), math.floor((z or 0) + 1e-6)
end

M.key = "station"

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

function M.match(obj)
    return tostring(obj and obj.name or "") == "penguin_station_point"
end

function M.build(ctx, obj, index)
    local x, z = ctx.helpers.center_from_obj(obj)
    local id = entity.spawn("BuilderStation" .. tostring(index), x, z, {"station", "facility", "builder_station"}, {
        collision = {
            enabled = false,
            layer = 1,
            mask = 0,
            shape = { type = "circle", radius = 0.35 },
        },
        visual = {
            display_name = "Station",
            color = {180, 100, 255, 255},
            visible = true,
        },
        properties = {
            delivered = false,
        },
    })

    if id then
        ctx.state.station_total = ctx.state.station_total + 1
    end

    return {
        key = M.key,
        index = index,
        entity_id = id,
        delivered = false,
    }
end

function M.on_player_enter(ctx, inst, px, py)
    if inst.delivered or not inst.entity_id then
        return
    end
    if ctx.state.package_count <= 0 then
        return
    end

    local s = entity.get(inst.entity_id)
    if not s or not s.alive then
        return
    end

    local sx, sz = ctx.helpers.grid_from_entity(s)
    if sx == px and sz == py then
        inst.delivered = true
        entity.set_property(inst.entity_id, "delivered", true)
        entity.set_color(inst.entity_id, 120, 220, 120, 255)
        ctx.state.package_count = ctx.state.package_count - 1
        ctx.state.station_delivered = ctx.state.station_delivered + 1
        ctx.state.score = ctx.state.score + 200
        ui.toast("投递成功 +200", 1.2)
    end
end

-- Drone gameplay: 查询驿站点位（station_id 可选）
function M.query_drone_station_points(ECS)
    local points = {}
    local objs = ECS.query_objects({
        component = "facility.station_point",
        names = { "penguin_station_point" },
    })
    if not objs or #objs == 0 then
        return points
    end

    for _, obj in ipairs(objs) do
        local ox, oz = obj_top_left(obj)
        local sid = nil
        if obj.properties and obj.properties.station_id then
            sid = tonumber(obj.properties.station_id)
            if sid ~= nil then
                sid = clamp(sid, 1, 10)
            end
        end
        table.insert(points, { id = sid, x = ox, z = oz })
    end

    return points
end

return M
