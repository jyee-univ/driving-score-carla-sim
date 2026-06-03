import argparse
import csv
import math
import os
import random
import sys
from pathlib import Path

import carla


def add_carla_agents_path():
    """
    CARLA pip package에는 agents 폴더가 없을 수 있다.
    따라서 WindowsNoEditor/PythonAPI/carla 경로를 sys.path에 추가한다.
    """
    possible_roots = [
        os.environ.get("CARLA_ROOT"),
        r"C:\CARLA\WindowsNoEditor",
        r"C:\CARLA_0.9.15\WindowsNoEditor",
        r"D:\CARLA\WindowsNoEditor",
    ]

    for root in possible_roots:
        if not root:
            continue

        agents_path = Path(root) / "PythonAPI" / "carla"
        if agents_path.exists():
            sys.path.append(str(agents_path))
            return True

    return False


add_carla_agents_path()

try:
    from agents.navigation.behavior_agent import BehaviorAgent
except Exception as e:
    BehaviorAgent = None
    print("[Warning] BehaviorAgent import failed.")
    print("CARLA_ROOT 환경변수 또는 CARLA 설치 경로를 확인하세요.")
    print("Error:", e)


def get_speed_kmh(vehicle: carla.Vehicle) -> float:
    v = vehicle.get_velocity()
    speed_ms = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
    return speed_ms * 3.6


def normalize_angle_diff_deg(current: float, previous: float) -> float:
    diff = current - previous

    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360

    return diff


def apply_driver_style(control: carla.VehicleControl, mode: str, tick: int) -> carla.VehicleControl:
    """
    BehaviorAgent의 기본 제어값에 운전자 성향을 추가로 반영한다.

    stable:
    - throttle 제한
    - brake 제한
    - steering 완만화

    aggressive:
    - 주기적으로 급가속, 급제동, 급조향 추가

    cautious:
    - 낮은 속도
    - 약한 brake 자주 사용
    """
    if mode == "stable":
        control.throttle = min(control.throttle, 0.35)
        control.brake = min(control.brake, 0.20)
        control.steer *= 0.75

    elif mode == "aggressive":
        phase = tick % 180

        if 20 <= phase < 55:
            control.throttle = max(control.throttle, 0.85)
            control.brake = 0.0

        elif 70 <= phase < 95:
            control.throttle = 0.0
            control.brake = max(control.brake, 0.65)

        elif 110 <= phase < 145:
            control.throttle = max(control.throttle, 0.55)
            control.steer += 0.22 if phase < 128 else -0.22

        control.steer = max(-1.0, min(1.0, control.steer))

    elif mode == "cautious":
        control.throttle = min(control.throttle, 0.25)

        phase = tick % 140
        if 40 <= phase < 55:
            control.brake = max(control.brake, 0.18)

        control.steer *= 0.85

    else:
        raise ValueError(f"Unknown driver mode: {mode}")

    return control


def make_agent(vehicle: carla.Vehicle, mode: str, target_speed: float):
    if BehaviorAgent is None:
        raise RuntimeError(
            "BehaviorAgent를 import할 수 없습니다. "
            "CARLA_ROOT를 WindowsNoEditor 경로로 설정하세요."
        )

    if mode == "stable":
        behavior = "normal"
    elif mode == "aggressive":
        behavior = "aggressive"
    elif mode == "cautious":
        behavior = "cautious"
    else:
        behavior = "normal"

    agent = BehaviorAgent(vehicle, behavior=behavior)
    agent.set_target_speed(target_speed)

    return agent


def choose_destination(world: carla.World, spawn_points: list, start_index: int = 0):
    start = spawn_points[start_index].location
    candidates = []

    for sp in spawn_points:
        if sp.location.distance(start) > 80:
            candidates.append(sp)

    if not candidates:
        return random.choice(spawn_points).location

    return random.choice(candidates).location


def calculate_basic_lane_offset(world_map, vehicle: carla.Vehicle) -> float:
    """
    차량 위치와 현재 waypoint 중심 사이의 거리로 lane offset을 간단 계산한다.
    정확한 좌우 부호까지는 단순화되어 있음.
    """
    loc = vehicle.get_location()
    waypoint = world_map.get_waypoint(loc, project_to_road=True)
    center = waypoint.transform.location

    dx = loc.x - center.x
    dy = loc.y - center.y

    return math.sqrt(dx ** 2 + dy ** 2)


def calculate_front_vehicle_distance(world: carla.World, ego_vehicle: carla.Vehicle, max_distance: float = 80.0):
    """
    ego 차량 전방에 있는 가장 가까운 차량과의 거리를 계산한다.
    단순 전방 벡터 기반이며, 같은 차선 판정은 추후 개선 가능.
    """
    ego_transform = ego_vehicle.get_transform()
    ego_loc = ego_transform.location
    forward = ego_transform.get_forward_vector()

    min_dist = None

    for actor in world.get_actors().filter("vehicle.*"):
        if actor.id == ego_vehicle.id:
            continue

        other_loc = actor.get_location()
        vec = other_loc - ego_loc

        forward_dot = vec.x * forward.x + vec.y * forward.y + vec.z * forward.z

        if forward_dot <= 0:
            continue

        dist = ego_loc.distance(other_loc)

        if dist <= max_distance:
            if min_dist is None or dist < min_dist:
                min_dist = dist

    return min_dist if min_dist is not None else max_distance


def calculate_headway(distance_m: float, speed_kmh: float) -> float:
    speed_ms = max(speed_kmh / 3.6, 1e-3)
    return distance_m / speed_ms


def calculate_ttc(distance_m: float, ego_speed_kmh: float, lead_speed_kmh: float = 0.0) -> float:
    ego_speed_ms = ego_speed_kmh / 3.6
    lead_speed_ms = lead_speed_kmh / 3.6
    relative_speed = ego_speed_ms - lead_speed_ms

    if relative_speed <= 0:
        return 999.0

    return distance_m / relative_speed


def calculate_simple_road_curvature(world_map, vehicle: carla.Vehicle) -> float:
    """
    현재 waypoint와 앞쪽 waypoint들의 yaw 변화량으로 도로 곡률을 간단 계산.
    """
    loc = vehicle.get_location()
    wp1 = world_map.get_waypoint(loc, project_to_road=True)
    next_wps = wp1.next(5.0)

    if not next_wps:
        return 0.0

    wp2 = next_wps[0]
    next_wps2 = wp2.next(5.0)

    if not next_wps2:
        return 0.0

    wp3 = next_wps2[0]

    yaw1 = math.radians(wp1.transform.rotation.yaw)
    yaw3 = math.radians(wp3.transform.rotation.yaw)

    dyaw = yaw3 - yaw1
    while dyaw > math.pi:
        dyaw -= 2 * math.pi
    while dyaw < -math.pi:
        dyaw += 2 * math.pi

    distance = wp1.transform.location.distance(wp3.transform.location)

    if distance <= 1e-6:
        return 0.0

    return abs(dyaw / distance)


def run_scenario(
    mode: str,
    output_csv: str,
    duration_s: float = 60.0,
    host: str = "localhost",
    port: int = 2000,
    target_speed: float | None = None,
):
    if target_speed is None:
        if mode == "stable":
            target_speed = 35.0
        elif mode == "aggressive":
            target_speed = 55.0
        elif mode == "cautious":
            target_speed = 25.0

    client = carla.Client(host, port)
    client.set_timeout(10.0)

    world = client.get_world()
    world_map = world.get_map()
    blueprint_library = world.get_blueprint_library()

    vehicle_bp = blueprint_library.filter("vehicle.*model3*")[0]
    spawn_points = world_map.get_spawn_points()

    if not spawn_points:
        raise RuntimeError("Spawn point를 찾을 수 없습니다.")

    spawn_index = 0
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[spawn_index])

    if vehicle is None:
        for i, sp in enumerate(spawn_points):
            vehicle = world.try_spawn_actor(vehicle_bp, sp)
            if vehicle is not None:
                spawn_index = i
                break

    if vehicle is None:
        raise RuntimeError("차량 spawn 실패")

    agent = make_agent(vehicle, mode, target_speed)
    destination = choose_destination(world, spawn_points, spawn_index)
    agent.set_destination(destination)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Mode: {mode}")
    print(f"[INFO] Target speed: {target_speed} km/h")
    print(f"[INFO] Output: {output_path}")

    prev_time = None
    prev_accel_x = None
    prev_accel_y = None
    prev_accel_z = None
    prev_yaw = None
    prev_steering_angle = None

    start_snapshot = world.get_snapshot()
    start_time = start_snapshot.timestamp.elapsed_seconds

    try:
        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame",
                "timestamp_s",
                "driver_type",
                "speed_kmh",
                "throttle",
                "brake",
                "steer_input",
                "steering_angle_deg",
                "gear",
                "reverse",
                "hand_brake",
                "accel_x_mps2",
                "accel_y_mps2",
                "accel_z_mps2",
                "gyro_z_radps",
                "yaw_rate_dps",
                "compass_yaw_deg",
                "steering_rate_degps",
                "jerk_x_mps3",
                "jerk_y_mps3",
                "jerk_z_mps3",
                "jerk_magnitude_mps3",
                "lane_offset_m",
                "front_vehicle_distance_m",
                "headway_s",
                "ttc_s",
                "road_curvature_1m",
            ])

            tick = 0

            while True:
                snapshot = world.get_snapshot()
                now = snapshot.timestamp.elapsed_seconds
                t = now - start_time

                if t > duration_s:
                    break

                if agent.done():
                    destination = choose_destination(world, spawn_points, spawn_index)
                    agent.set_destination(destination)

                control = agent.run_step()
                control.manual_gear_shift = False
                control = apply_driver_style(control, mode, tick)

                vehicle.apply_control(control)

                speed_kmh = get_speed_kmh(vehicle)
                acceleration = vehicle.get_acceleration()
                ax, ay, az = acceleration.x, acceleration.y, acceleration.z

                transform = vehicle.get_transform()
                yaw = transform.rotation.yaw

                if prev_time is None:
                    dt = None
                else:
                    dt = max(now - prev_time, 1e-6)

                steering_angle_deg = control.steer * 70.0

                if dt is None:
                    steering_rate = 0.0
                    yaw_rate_dps = 0.0
                    gyro_z_radps = 0.0
                    jerk_x = 0.0
                    jerk_y = 0.0
                    jerk_z = 0.0
                else:
                    steering_rate = (steering_angle_deg - prev_steering_angle) / dt

                    yaw_diff = normalize_angle_diff_deg(yaw, prev_yaw)
                    yaw_rate_dps = yaw_diff / dt
                    gyro_z_radps = math.radians(yaw_rate_dps)

                    jerk_x = (ax - prev_accel_x) / dt
                    jerk_y = (ay - prev_accel_y) / dt
                    jerk_z = (az - prev_accel_z) / dt

                jerk_mag = math.sqrt(jerk_x ** 2 + jerk_y ** 2 + jerk_z ** 2)

                lane_offset = calculate_basic_lane_offset(world_map, vehicle)
                front_dist = calculate_front_vehicle_distance(world, vehicle)
                headway = calculate_headway(front_dist, speed_kmh)
                ttc = calculate_ttc(front_dist, speed_kmh)
                road_curvature = calculate_simple_road_curvature(world_map, vehicle)

                writer.writerow([
                    snapshot.frame,
                    round(t, 3),
                    mode,
                    round(speed_kmh, 3),
                    round(control.throttle, 4),
                    round(control.brake, 4),
                    round(control.steer, 4),
                    round(steering_angle_deg, 3),
                    control.gear,
                    control.reverse,
                    control.hand_brake,
                    round(ax, 5),
                    round(ay, 5),
                    round(az, 5),
                    round(gyro_z_radps, 5),
                    round(yaw_rate_dps, 5),
                    round(yaw, 5),
                    round(steering_rate, 5),
                    round(jerk_x, 5),
                    round(jerk_y, 5),
                    round(jerk_z, 5),
                    round(jerk_mag, 5),
                    round(lane_offset, 5),
                    round(front_dist, 5),
                    round(headway, 5),
                    round(ttc, 5),
                    round(road_curvature, 8),
                ])

                prev_time = now
                prev_accel_x = ax
                prev_accel_y = ay
                prev_accel_z = az
                prev_yaw = yaw
                prev_steering_angle = steering_angle_deg

                tick += 1

    finally:
        vehicle.destroy()
        print("[INFO] Vehicle destroyed.")
        print(f"[INFO] Saved CSV: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["stable", "aggressive", "cautious"],
        default="stable",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--target-speed", type=float, default=None)

    args = parser.parse_args()

    if args.output is None:
        args.output = f"data/sample/carla_{args.mode}_signals.csv"

    run_scenario(
        mode=args.mode,
        output_csv=args.output,
        duration_s=args.duration,
        host=args.host,
        port=args.port,
        target_speed=args.target_speed,
    )


if __name__ == "__main__":
    main()