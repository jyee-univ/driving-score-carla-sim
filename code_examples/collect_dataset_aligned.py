#!/usr/bin/env python3
"""
collect_dataset_aligned.py

CARLA 0.9.16용 차량 신호 수집기
- manual_control.py가 생성한 hero 차량을 자동으로 찾습니다.
- 기존 manual_control.py 파일은 수정하지 않습니다.
- UAH-DriveSet 계열과 CAN/OBD 계열 분석에 필요한 핵심 신호만 CSV로 저장합니다.
- Ctrl+C를 누르면 저장을 마치고 종료합니다.

주의:
- CARLA 기본 Python API는 실시간 엔진 RPM을 직접 제공하지 않습니다.
- 이 파일은 RPM을 임의로 만들어 저장하지 않습니다.
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
    # 공통 식별 정보
    "frame",
    "timestamp_s",

    # CAN/OBD 대응 차량 내부 신호
    "speed_kmh",
    "throttle",
    "brake",
    "steer_input",
    "steering_angle_deg",
    "gear",
    "reverse",
    "hand_brake",

    # UAH 대응 스마트폰형 IMU 신호
    "accel_x_mps2",
    "accel_y_mps2",
    "accel_z_mps2",
    "gyro_x_radps",
    "gyro_y_radps",
    "gyro_z_radps",
    "compass_rad",

    # UAH 대응 GNSS 신호
    "latitude",
    "longitude",
    "altitude_m",

    # 위 원본 신호로부터 계산한 파생 신호
    "steering_rate_degps",
    "jerk_x_mps3",
    "jerk_y_mps3",
    "jerk_z_mps3",
    "jerk_magnitude_mps3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "manual_control.py가 생성한 hero 차량의 UAH/CAN-OBD 대응 신호를 "
            "CSV로 저장합니다."
        )
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="CARLA 서버 주소. 기본값: 127.0.0.1",
    )
    parser.add_argument(
        "-p",
        "--port",
        default=2000,
        type=int,
        help="CARLA 서버 TCP 포트. 기본값: 2000",
    )
    parser.add_argument(
        "--hz",
        default=20.0,
        type=float,
        help="IMU 및 GNSS 수집 주파수. 기본값: 20 Hz",
    )
    parser.add_argument(
        "--label",
        default="manual",
        help="주행 종류 라벨. 예: normal, aggressive, drowsy. 기본값: manual",
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
    return parser.parse_args()


def magnitude3(x: float, y: float, z: float) -> float:
    return math.sqrt(x * x + y * y + z * z)


def find_hero_vehicle(world: carla.World) -> Optional[carla.Vehicle]:
    """manual_control.py가 생성한 role_name=hero 차량을 찾습니다."""
    vehicles = world.get_actors().filter("vehicle.*")

    # manual_control.py는 일반적으로 hero 역할명을 사용합니다.
    for vehicle in vehicles:
        if vehicle.attributes.get("role_name", "") == "hero":
            return vehicle

    # 다른 실습 스크립트에서 ego라는 역할명을 쓸 가능성도 고려합니다.
    for vehicle in vehicles:
        if vehicle.attributes.get("role_name", "") == "ego":
            return vehicle

    return None


def wait_for_hero_vehicle(
    world: carla.World,
    timeout_seconds: float,
) -> carla.Vehicle:
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
    """
    실제 물리 기반 앞바퀴 조향각을 읽습니다.
    차량에 따라 조회가 실패할 수 있으므로 실패 시 NaN을 반환합니다.
    """
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
    sensor_tick: float,
) -> carla.Sensor:
    blueprint = world.get_blueprint_library().find(blueprint_id)
    blueprint.set_attribute("sensor_tick", f"{sensor_tick:.6f}")

    return world.spawn_actor(
        blueprint,
        carla.Transform(),
        attach_to=vehicle,
        attachment_type=carla.AttachmentType.Rigid,
    )


def safe_put(q: "queue.Queue[carla.IMUMeasurement]", data: carla.IMUMeasurement) -> None:
    """큐가 꽉 차면 가장 오래된 항목을 하나 버리고 최신 데이터를 넣습니다."""
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
    return run_dir / "carla_signals.csv"


def main() -> None:
    args = parse_args()

    if args.hz <= 0:
        raise ValueError("--hz는 0보다 커야 합니다.")

    sensor_tick = 1.0 / args.hz

    print("[연결] CARLA 서버에 연결 중...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    print("[검색] manual_control.py가 생성한 hero 차량을 찾습니다.")
    vehicle = wait_for_hero_vehicle(world, args.wait_seconds)

    print(f"[차량] ID={vehicle.id}, type={vehicle.type_id}")
    print(f"[설정] 수집 주파수={args.hz:.1f} Hz")

    output_path = make_output_path(args.output_root, args.label)
    print(f"[저장] {output_path.resolve()}")

    imu_queue: "queue.Queue[carla.IMUMeasurement]" = queue.Queue(maxsize=1000)
    gnss_lock = threading.Lock()
    latest_gnss: dict[str, float] = {
        "latitude": float("nan"),
        "longitude": float("nan"),
        "altitude_m": float("nan"),
    }

    imu_sensor: Optional[carla.Sensor] = None
    gnss_sensor: Optional[carla.Sensor] = None

    previous_timestamp: Optional[float] = None
    previous_steering_angle: Optional[float] = None
    previous_accel: Optional[tuple[float, float, float]] = None
    row_count = 0

    def on_gnss(data: carla.GnssMeasurement) -> None:
        nonlocal latest_gnss
        with gnss_lock:
            latest_gnss = {
                "latitude": float(data.latitude),
                "longitude": float(data.longitude),
                "altitude_m": float(data.altitude),
            }

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

        imu_sensor.listen(lambda data: safe_put(imu_queue, data))
        gnss_sensor.listen(on_gnss)

        with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()

            print("")
            print("[수집 시작] Pygame 창에서 차량을 운전하세요.")
            print("[수집 종료] 이 PowerShell 창에서 Ctrl+C를 누르세요.")
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
                accel = imu_data.accelerometer
                gyro = imu_data.gyroscope

                velocity = vehicle.get_velocity()
                speed_kmh = 3.6 * magnitude3(
                    float(velocity.x),
                    float(velocity.y),
                    float(velocity.z),
                )

                control = vehicle.get_control()
                steering_angle_deg = get_average_front_wheel_angle_deg(vehicle)

                steering_rate_degps = float("nan")
                jerk_x = float("nan")
                jerk_y = float("nan")
                jerk_z = float("nan")
                jerk_magnitude = float("nan")

                if previous_timestamp is not None:
                    dt = current_timestamp - previous_timestamp

                    if dt > 0:
                        if (
                            previous_steering_angle is not None
                            and math.isfinite(steering_angle_deg)
                            and math.isfinite(previous_steering_angle)
                        ):
                            steering_rate_degps = (
                                steering_angle_deg - previous_steering_angle
                            ) / dt

                        if previous_accel is not None:
                            jerk_x = (float(accel.x) - previous_accel[0]) / dt
                            jerk_y = (float(accel.y) - previous_accel[1]) / dt
                            jerk_z = (float(accel.z) - previous_accel[2]) / dt
                            jerk_magnitude = magnitude3(jerk_x, jerk_y, jerk_z)

                with gnss_lock:
                    gnss = dict(latest_gnss)

                row = {
                    "frame": int(imu_data.frame),
                    "timestamp_s": current_timestamp,

                    "speed_kmh": speed_kmh,
                    "throttle": float(control.throttle),
                    "brake": float(control.brake),
                    "steer_input": float(control.steer),
                    "steering_angle_deg": steering_angle_deg,
                    "gear": int(control.gear),
                    "reverse": int(bool(control.reverse)),
                    "hand_brake": int(bool(control.hand_brake)),

                    "accel_x_mps2": float(accel.x),
                    "accel_y_mps2": float(accel.y),
                    "accel_z_mps2": float(accel.z),
                    "gyro_x_radps": float(gyro.x),
                    "gyro_y_radps": float(gyro.y),
                    "gyro_z_radps": float(gyro.z),
                    "compass_rad": float(imu_data.compass),

                    "latitude": gnss["latitude"],
                    "longitude": gnss["longitude"],
                    "altitude_m": gnss["altitude_m"],

                    "steering_rate_degps": steering_rate_degps,
                    "jerk_x_mps3": jerk_x,
                    "jerk_y_mps3": jerk_y,
                    "jerk_z_mps3": jerk_z,
                    "jerk_magnitude_mps3": jerk_magnitude,
                }

                writer.writerow(row)
                row_count += 1

                # 갑작스러운 종료에도 최대한 많은 행이 저장되도록 주기적으로 flush합니다.
                if row_count % 20 == 0:
                    csv_file.flush()

                if row_count % 100 == 0:
                    print(
                        f"[수집 중] {row_count}행 저장 | "
                        f"t={current_timestamp:.2f}s | "
                        f"speed={speed_kmh:.1f} km/h"
                    )

                previous_timestamp = current_timestamp
                previous_steering_angle = steering_angle_deg
                previous_accel = (
                    float(accel.x),
                    float(accel.y),
                    float(accel.z),
                )

    except KeyboardInterrupt:
        print("")
        print("[종료] Ctrl+C 입력을 받았습니다.")

    finally:
        for sensor in (imu_sensor, gnss_sensor):
            if sensor is not None:
                try:
                    sensor.stop()
                except Exception:
                    pass

                try:
                    sensor.destroy()
                except Exception:
                    pass

        print(f"[완료] 총 {row_count}행을 저장했습니다.")
        print(f"[파일] {output_path.resolve()}")


if __name__ == "__main__":
    main()
