# 기존 CARLA 데이터 수집 파일에 추가
from curvature_utils import get_avg_road_curvature

#수집 루프 안에 추가
carla_map = world.get_map()

road_curvature_1m = get_avg_road_curvature(
    carla_map,
    vehicle,
    step=2.0,
    n=5
)

#csv 저장 row 안에 추가
row = {
    "frame": frame,
    "timestamp_s": timestamp_s,
    "speed_kmh": speed_kmh,
    "throttle": control.throttle,
    "brake": control.brake,
    "steer_input": control.steer,
    "steering_angle_deg": steering_angle_deg,
    "gear": control.gear,
    "reverse": control.reverse,
    "hand_brake": control.hand_brake,
    "accel_x_mps2": accel_x,
    "accel_y_mps2": accel_y,
    "accel_z_mps2": accel_z,
    "gyro_x_radps": gyro_x,
    "gyro_y_radps": gyro_y,
    "gyro_z_radps": gyro_z,
    "compass_rad": compass_rad,
    "latitude": latitude,
    "longitude": longitude,
    "altitude_m": altitude_m,
    "steering_rate_degps": steering_rate_degps,
    "jerk_x_mps3": jerk_x,
    "jerk_y_mps3": jerk_y,
    "jerk_z_mps3": jerk_z,
    "jerk_magnitude_mps3": jerk_magnitude,

    # 추가된 값
    "road_curvature_1m": road_curvature_1m,
}
