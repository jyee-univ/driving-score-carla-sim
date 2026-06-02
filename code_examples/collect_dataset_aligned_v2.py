#!/usr/bin/env python3
"""
collect_dataset_aligned_v2.py

CARLA 0.9.16용 차량 신호 수집기 개선판
- manual_control.py가 생성한 role_name=hero 차량을 자동으로 찾습니다.
- 기존 manual_control.py 파일은 수정하지 않습니다.
- UAH-DriveSet 및 CAN/OBD 분석에 대응되는 핵심 신호를 CSV로 저장합니다.
- 초기 센서 부착 직후의 비정상적인 IMU 과도응답을 warm-up 구간으로 제외합니다.
- 원본 IMU 값은 그대로 보존하고, 분석에 더 안정적인 속도 기반 종방향 가속도와 jerk를 함께 저장합니다.
- 충돌 이벤트와 IMU 이상치 플래그를 품질 확인용으로 추가합니다.
- Ctrl+C를 누르면 안전하게 저장을 마치고 종료합니다.

중요:
- CARLA 기본 Python API는 실시간 엔진 RPM을 직접 제공하지 않습니다.
- rpm을 실제 측정값처럼 임의 생성하지 않습니다.
- 수집 중 manual_control.py에서 Backspace로 차량을 바꾸지 마세요.
"""

from __future__ import annotations

import argparse
import csv
import math
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import carla


CSV_FIELDS = [
    # 공통 시간축
    "frame",
    "timestamp_s",
    "relative_time_s",

    # CAN/OBD 대응 차량 내부 신호
    "speed_kmh",
    "speed_mps",
    "throttle",
    "brake",
    "steer_input",
    "steering_angle_deg",
    "steering_rate_degps",
    "gear",
    "reverse",
    "hand_brake",

    # UAH 대응 스마트폰형 IMU 원본 신호
    "accel_x_raw_mps2",
    "accel_y_raw_mps2",
    "accel_z_raw_mps2",
    "accel_magnitude_raw_mps2",
    "gyro_x_radps",
    "gyro_y_radps",
    "gyro_z_radps",
    "compass_rad",

    # UAH 대응 GNSS 신호
    "latitude",
    "longitude",
    "altitude_m",

    # 속도 시계열에서 계산한 분석용 종방향 신호
    "longitudinal_accel_from_speed_mps2",
    "longitudinal_jerk_from_speed_mps3",

    # 품질 확인용 플래그
    "imu_outlier_flag",
    "collision_flag",
    "collision_impulse",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "manual_control.py가 만든 hero 차량의 UAH/CAN-OBD 대응 신호를 "
            "CSV로 저장합니다."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("-p", "--port", default=2000, type=int)
    parser.add_argument(
        "--hz",
        default=20.0,
        type=float,
        help="IMU 및 GNSS 수집 주파수. 기본값: 20 Hz",
    )
    parser.add_argument(
        "--label",
        default="manual",
        help="주행 종류 라벨. 예: normal, aggressive, drowsy",
    )
    parser.add_argument(
        "--output-root",
        default="CARLA_DATA",
        help="결과 저장 상위 폴더. 기본값: CARLA_DATA",
    )
    parser.add_argument(
        "--wait-seconds",
        default=60.0,
        type=float,
        help="hero 차량을 기다리는 최대 시간. 기본값: 60초",
    )
    parser.add_argument(
        "--warmup-seconds",
        default=2.0,
        type=float,
        help="센서 부착 직후 저장하지 않을 시간. 기본값: 2초",
    )
    parser.add_argument(
        "--imu-outlier-threshold",
        default=50.0,
        type=float,
        help="IMU 가속도 크기 이상치 기준(m/s^2). 기본값: 50",
    )
    return parser.parse_args()


def magnitude3(x: float, y: float, z: float) -> float:
    return math.sqrt(x * x + y * y + z * z)


def find_hero_vehicle(world: carla.World) -> Optional[carla.Vehicle]:
    vehicles = world.get_actors().filter("vehicle.*")

    for vehicle in vehicles:
        if vehicle.attributes.get("role_name", "") == "hero":
            return vehicle

    for vehicle in vehicles:
        if vehicle.attributes.get("role_name", "") == "ego":
            return vehicle

    return None


def wait_for_hero_vehicle(world: carla.World, timeout_seconds: float) -> carla.Vehicle:
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        vehicle = find_hero_vehicle(world)
        if vehicle is not None:
            return vehicle

        print(
            "[대기] hero 차량을 찾는 중입니다. "
            "다른 PowerShell에서 manual_control.py를 먼저 실행하세요."
        )
        time.sleep(1.0)

    raise RuntimeError(
        f"{timeout_seconds:.0f}초 동안 hero 차량을 찾지 못했습니다. "
        "manual_control.py 실행 여부를 확인하세요."
    )


def get_average_front_wheel_angle_deg(vehicle: carla.Vehicle) -> float:
    try:
        fl = vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel)
        fr = vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FR_Wheel)
        return 0.5 * (float(fl) + float(fr))
    except Exception:
        return float("nan")


def spawn_sensor(
    world: carla.World,
    vehicle: carla.Vehicle,
    blueprint_id: str,
    sensor_tick: Optional[float] = None,
) -> carla.Sensor:
    blueprint = world.get_blueprint_library().find(blueprint_id)

    if sensor_tick is not None and blueprint.has_attribute("sensor_tick"):
        blueprint.set_attribute("sensor_tick", f"{sensor_tick:.6f}")

    return world.spawn_actor(
        blueprint,
        carla.Transform(),
        attach_to=vehicle,
        attachment_type=carla.AttachmentType.Rigid,
    )


def safe_put(q: "queue.Queue[carla.IMUMeasurement]", data: carla.IMUMeasurement) -> None:
    try:
        q.put_nowait(data)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        q.put_nowait(data)


def make_output_path(output_root: str, label: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"run_{timestamp}_{label}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir / "carla_signals_v2.csv"


def main() -> None:
    args = parse_args()

    if args.hz <= 0:
        raise ValueError("--hz는 0보다 커야 합니다.")

    if args.warmup_seconds < 0:
        raise ValueError("--warmup-seconds는 0 이상이어야 합니다.")

    if args.imu_outlier_threshold <= 0:
        raise ValueError("--imu-outlier-threshold는 0보다 커야 합니다.")

    sensor_tick = 1.0 / args.hz

    print("[연결] CARLA 서버에 연결 중...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    print("[검색] manual_control.py가 생성한 hero 차량을 찾습니다.")
    vehicle = wait_for_hero_vehicle(world, args.wait_seconds)

    print(f"[차량] ID={vehicle.id}, type={vehicle.type_id}")
    print(f"[설정] 수집 주파수={args.hz:.1f} Hz")
    print(f"[설정] warm-up 제외 시간={args.warmup_seconds:.1f}초")

    output_path = make_output_path(args.output_root, args.label)
    print(f"[저장] {output_path.resolve()}")

    imu_queue: "queue.Queue[carla.IMUMeasurement]" = queue.Queue(maxsize=1000)

    gnss_lock = threading.Lock()
    latest_gnss = {
        "latitude": float("nan"),
        "longitude": float("nan"),
        "altitude_m": float("nan"),
    }

    collision_lock = threading.Lock()
    collision_since_last_row = {
        "flag": 0,
        "max_impulse": 0.0,
    }

    imu_sensor: Optional[carla.Sensor] = None
    gnss_sensor: Optional[carla.Sensor] = None
    collision_sensor: Optional[carla.Sensor] = None

    first_sensor_timestamp: Optional[float] = None
    first_saved_timestamp: Optional[float] = None
    previous_timestamp: Optional[float] = None
    previous_speed_mps: Optional[float] = None
    previous_longitudinal_accel: Optional[float] = None
    previous_steering_angle: Optional[float] = None

    row_count = 0
    skipped_warmup_count = 0

    def on_gnss(data: carla.GnssMeasurement) -> None:
        nonlocal latest_gnss
        with gnss_lock:
            latest_gnss = {
                "latitude": float(data.latitude),
                "longitude": float(data.longitude),
                "altitude_m": float(data.altitude),
            }

    def on_collision(data: carla.CollisionEvent) -> None:
        impulse = data.normal_impulse
        intensity = magnitude3(
            float(impulse.x),
            float(impulse.y),
            float(impulse.z),
        )

        with collision_lock:
            collision_since_last_row["flag"] = 1
            collision_since_last_row["max_impulse"] = max(
                collision_since_last_row["max_impulse"],
                intensity,
            )

    try:
        imu_sensor = spawn_sensor(
            world,
            vehicle,
            blueprint_id="sensor.other.imu",
            sensor_tick=sensor_tick,
        )
        gnss_sensor = spawn_sensor(
            world,
            vehicle,
            blueprint_id="sensor.other.gnss",
            sensor_tick=sensor_tick,
        )
        collision_sensor = spawn_sensor(
            world,
            vehicle,
            blueprint_id="sensor.other.collision",
        )

        imu_sensor.listen(lambda data: safe_put(imu_queue, data))
        gnss_sensor.listen(on_gnss)
        collision_sensor.listen(on_collision)

        with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()

            print("")
            print("[수집 준비] 센서 초기 과도응답을 제외한 뒤 저장을 시작합니다.")
            print("[운전] Pygame 창에서 차량을 운전하세요.")
            print("[종료] 이 PowerShell 창에서 Ctrl+C를 누르세요.")
            print("[주의] 수집 중에는 Backspace로 차량을 바꾸지 마세요.")
            print("")

            while True:
                if not vehicle.is_alive:
                    raise RuntimeError(
                        "hero 차량이 사라졌습니다. "
                        "manual_control.py에서 차량을 바꿨다면 수집기를 다시 실행하세요."
                    )

                try:
                    imu_data = imu_queue.get(timeout=2.0)
                except queue.Empty:
                    print("[경고] IMU 데이터가 2초 동안 들어오지 않았습니다.")
                    continue

                current_timestamp = float(imu_data.timestamp)

                if first_sensor_timestamp is None:
                    first_sensor_timestamp = current_timestamp

                if current_timestamp - first_sensor_timestamp < args.warmup_seconds:
                    skipped_warmup_count += 1
                    continue

                if first_saved_timestamp is None:
                    first_saved_timestamp = current_timestamp
                    print("[수집 시작] warm-up 제외 완료. CSV 기록을 시작합니다.")

                velocity = vehicle.get_velocity()
                speed_mps = magnitude3(
                    float(velocity.x),
                    float(velocity.y),
                    float(velocity.z),
                )
                speed_kmh = 3.6 * speed_mps

                control = vehicle.get_control()
                steering_angle_deg = get_average_front_wheel_angle_deg(vehicle)

                accel = imu_data.accelerometer
                gyro = imu_data.gyroscope

                accel_x = float(accel.x)
                accel_y = float(accel.y)
                accel_z = float(accel.z)
                accel_magnitude = magnitude3(accel_x, accel_y, accel_z)
                imu_outlier_flag = int(accel_magnitude > args.imu_outlier_threshold)

                steering_rate = float("nan")
                longitudinal_accel = float("nan")
                longitudinal_jerk = float("nan")

                if previous_timestamp is not None:
                    dt = current_timestamp - previous_timestamp

                    if dt > 0:
                        if (
                            previous_steering_angle is not None
                            and math.isfinite(steering_angle_deg)
                            and math.isfinite(previous_steering_angle)
                        ):
                            steering_rate = (
                                steering_angle_deg - previous_steering_angle
                            ) / dt

                        if previous_speed_mps is not None:
                            longitudinal_accel = (
                                speed_mps - previous_speed_mps
                            ) / dt

                        if (
                            previous_longitudinal_accel is not None
                            and math.isfinite(longitudinal_accel)
                        ):
                            longitudinal_jerk = (
                                longitudinal_accel - previous_longitudinal_accel
                            ) / dt

                with gnss_lock:
                    gnss = dict(latest_gnss)

                with collision_lock:
                    collision_flag = int(collision_since_last_row["flag"])
                    collision_impulse = float(
                        collision_since_last_row["max_impulse"]
                    )
                    collision_since_last_row["flag"] = 0
                    collision_since_last_row["max_impulse"] = 0.0

                row = {
                    "frame": int(imu_data.frame),
                    "timestamp_s": current_timestamp,
                    "relative_time_s": current_timestamp - first_saved_timestamp,

                    "speed_kmh": speed_kmh,
                    "speed_mps": speed_mps,
                    "throttle": float(control.throttle),
                    "brake": float(control.brake),
                    "steer_input": float(control.steer),
                    "steering_angle_deg": steering_angle_deg,
                    "steering_rate_degps": steering_rate,
                    "gear": int(control.gear),
                    "reverse": int(bool(control.reverse)),
                    "hand_brake": int(bool(control.hand_brake)),

                    "accel_x_raw_mps2": accel_x,
                    "accel_y_raw_mps2": accel_y,
                    "accel_z_raw_mps2": accel_z,
                    "accel_magnitude_raw_mps2": accel_magnitude,
                    "gyro_x_radps": float(gyro.x),
                    "gyro_y_radps": float(gyro.y),
                    "gyro_z_radps": float(gyro.z),
                    "compass_rad": float(imu_data.compass),

                    "latitude": gnss["latitude"],
                    "longitude": gnss["longitude"],
                    "altitude_m": gnss["altitude_m"],

                    "longitudinal_accel_from_speed_mps2": longitudinal_accel,
                    "longitudinal_jerk_from_speed_mps3": longitudinal_jerk,

                    "imu_outlier_flag": imu_outlier_flag,
                    "collision_flag": collision_flag,
                    "collision_impulse": collision_impulse,
                }

                writer.writerow(row)
                row_count += 1

                if row_count % 20 == 0:
                    csv_file.flush()

                if row_count % 100 == 0:
                    print(
                        f"[수집 중] {row_count}행 | "
                        f"t={row['relative_time_s']:.1f}s | "
                        f"speed={speed_kmh:.1f} km/h | "
                        f"imu_outlier={imu_outlier_flag} | "
                        f"collision={collision_flag}"
                    )

                previous_timestamp = current_timestamp
                previous_speed_mps = speed_mps
                previous_longitudinal_accel = longitudinal_accel
                previous_steering_angle = steering_angle_deg

    except KeyboardInterrupt:
        print("")
        print("[종료] Ctrl+C 입력을 받았습니다.")

    finally:
        for sensor in (imu_sensor, gnss_sensor, collision_sensor):
            if sensor is not None:
                try:
                    sensor.stop()
                except Exception:
                    pass

                try:
                    sensor.destroy()
                except Exception:
                    pass

        print(f"[완료] warm-up 구간 {skipped_warmup_count}행 제외")
        print(f"[완료] 총 {row_count}행 저장")
        print(f"[파일] {output_path.resolve()}")


if __name__ == "__main__":
    main()
