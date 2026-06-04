import numpy as np
import pandas as pd


def ratio(condition):
    return np.mean(condition.astype(float))


def compute_driver_scores(df):
    df = df.copy()

    # =========================
    # 1. 기본 값 읽기
    # =========================
    speed_mps = df["speed_kmh"] / 3.6

    accel_x = df["accel_x_mps2"]
    lateral_accel = df["accel_y_mps2"]

    throttle = df["throttle"]
    brake = df["brake"]

    steering_rate = df["steering_rate_degps"]

    actual_yaw_rate_dps = np.abs(
        df["gyro_z_radps"] * 180 / np.pi
    )

    if "jerk_magnitude_mps3" in df.columns:
        jerk = df["jerk_magnitude_mps3"]
    else:
        jerk = df["jerk_magnitude_mps"]

    # =========================
    # 2. 도로 곡률값 읽기
    # =========================
    road_curvature = df["road_curvature_1m"].fillna(0)

    # =========================
    # 3. 곡률 보정값 생성
    # =========================
    curve_ref = 0.05

    curve_factor = np.clip(
        road_curvature / curve_ref,
        0,
        1
    )

    curve_discount = 1 - 0.4 * curve_factor

    curve_discount = np.clip(
        curve_discount,
        0.6,
        1.0
    )

    # =========================
    # 4. 도로 때문에 자연스럽게 생기는 값 계산
    # =========================
    expected_yaw_rate_dps = (
        speed_mps
        * road_curvature
        * 180
        / np.pi
    )

    expected_lateral_accel = (
        speed_mps ** 2
        * road_curvature
    )

    # =========================
    # 5. 도로 곡률로 설명 안 되는 초과 움직임 계산
    # =========================
    excess_yaw_rate_dps = np.maximum(
        actual_yaw_rate_dps - expected_yaw_rate_dps,
        0
    )

    actual_lateral_accel = np.abs(lateral_accel)

    excess_lateral_accel = np.maximum(
        actual_lateral_accel - expected_lateral_accel,
        0
    )

    # =========================
    # 6. 기본 이벤트 계산
    # =========================
    harsh_accel_event = accel_x > 2.5
    harsh_brake_event = accel_x < -3.0
    hard_brake_input_event = brake > 0.6

    high_jerk_event = jerk > 5.0

    throttle_spike_event = throttle.diff().abs().fillna(0) > 0.25
    brake_usage_event = brake > 0.1

    # =========================
    # 7. Safety Score 계산
    # =========================
    harsh_accel_rate = ratio(harsh_accel_event)
    harsh_brake_rate = ratio(harsh_brake_event)
    hard_brake_input_rate = ratio(hard_brake_input_event)

    high_yaw_rate_ratio = ratio(
        excess_yaw_rate_dps > 15
    )

    high_lateral_accel_ratio = ratio(
        excess_lateral_accel > 1.5
    )

    high_steering_rate_event = (
        np.abs(steering_rate) > 80
    )

    high_steering_rate_ratio = np.mean(
        high_steering_rate_event.astype(float)
        * curve_discount
    )

    high_jerk_ratio = ratio(high_jerk_event)

    safety_score = (
        100
        - 15 * harsh_accel_rate
        - 20 * harsh_brake_rate
        - 15 * hard_brake_input_rate
        - 15 * high_yaw_rate_ratio
        - 15 * high_lateral_accel_ratio
        - 10 * high_steering_rate_ratio
        - 10 * high_jerk_ratio
    )

    # =========================
    # 8. Smoothness Score 계산
    # =========================
    jerk_risk = np.clip(
        jerk.mean() / 5.0,
        0,
        1
    )

    curve_adjusted_steering_rate = (
        np.abs(steering_rate)
        * curve_discount
    )

    steering_rate_risk = np.clip(
        curve_adjusted_steering_rate.mean() / 100,
        0,
        1
    )

    yaw_rate_variation_risk = np.clip(
        excess_yaw_rate_dps.std() / 20,
        0,
        1
    )

    curve_adjusted_accel = (
        accel_x
        * (1 - 0.3 * curve_factor)
    )

    acceleration_variation_risk = np.clip(
        curve_adjusted_accel.std() / 2.5,
        0,
        1
    )

    fft_high_frequency_risk = 0
    stft_event_risk = 0

    smoothness_score = (
        100
        - 25 * jerk_risk
        - 20 * steering_rate_risk
        - 15 * yaw_rate_variation_risk
        - 15 * acceleration_variation_risk
        - 15 * fft_high_frequency_risk
        - 10 * stft_event_risk
    )

    # =========================
    # 9. Eco-like Score 계산
    # =========================
    throttle_spike_rate = ratio(throttle_spike_event)

    brake_usage_rate = np.mean(
        brake_usage_event.astype(float)
        * (1 - 0.3 * curve_factor)
    )

    harsh_accel_rate_eco = harsh_accel_rate

    harsh_brake_rate_eco = np.mean(
        harsh_brake_event.astype(float)
        * (1 - 0.3 * curve_factor)
    )

    speed_diff = df["speed_kmh"].diff().abs().fillna(0)

    speed_instability_risk = np.clip(
        (
            speed_diff
            * (1 - 0.3 * curve_factor)
        ).mean() / 5,
        0,
        1
    )

    eco_score = (
        100
        - 25 * throttle_spike_rate
        - 20 * brake_usage_rate
        - 20 * harsh_accel_rate_eco
        - 20 * harsh_brake_rate_eco
        - 15 * speed_instability_risk
    )

    # =========================
    # 10. 점수 범위 제한
    # =========================
    safety_score = np.clip(safety_score, 0, 100)
    smoothness_score = np.clip(smoothness_score, 0, 100)
    eco_score = np.clip(eco_score, 0, 100)

    return {
        "Safety Score": round(safety_score, 2),
        "Smoothness Score": round(smoothness_score, 2),
        "Eco-like Score": round(eco_score, 2),

        "Harsh Accel Rate": round(harsh_accel_rate, 4),
        "Harsh Brake Rate": round(harsh_brake_rate, 4),
        "Hard Brake Input Rate": round(hard_brake_input_rate, 4),

        "High Yaw Rate Ratio": round(high_yaw_rate_ratio, 4),
        "High Lateral Accel Ratio": round(high_lateral_accel_ratio, 4),
        "High Steering Rate Ratio": round(high_steering_rate_ratio, 4),
        "High Jerk Ratio": round(high_jerk_ratio, 4),

        "Jerk Risk": round(jerk_risk, 4),
        "Steering Rate Risk": round(steering_rate_risk, 4),
        "Yaw Rate Variation Risk": round(yaw_rate_variation_risk, 4),
        "Acceleration Variation Risk": round(acceleration_variation_risk, 4),

        "Throttle Spike Rate": round(throttle_spike_rate, 4),
        "Brake Usage Rate": round(brake_usage_rate, 4),
        "Harsh Brake Rate Eco": round(harsh_brake_rate_eco, 4),
        "Speed Instability Risk": round(speed_instability_risk, 4),

        "Mean Road Curvature": round(road_curvature.mean(), 5),
        "Mean Curve Factor": round(curve_factor.mean(), 4),
    }
