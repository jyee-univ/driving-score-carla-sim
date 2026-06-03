import numpy as np
import pandas as pd


def ratio_over_threshold(series: pd.Series, threshold: float, use_abs: bool = True) -> float:
    x = series.dropna().to_numpy(dtype=float)

    if len(x) == 0:
        return 0.0

    if use_abs:
        x = np.abs(x)

    return float(np.mean(x >= threshold))


def normalize_risk(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0

    return float(np.clip(value / max_value, 0, 1))


def calculate_safety_score(df: pd.DataFrame) -> tuple[float, dict]:
    """
    CARLA 기본 CSV 기준 Safety Score.
    현재는 lane_offset, TTC, front_distance가 없을 수 있으므로
    급가속, 급감속, 급제동, 급조향, yaw/lateral 위험 중심으로 계산한다.
    """
    harsh_accel = 0.0
    harsh_brake = 0.0

    if "longitudinal_accel_mps2" in df.columns:
        harsh_accel = float(np.mean(df["longitudinal_accel_mps2"] >= 2.5))
        harsh_brake = float(np.mean(df["longitudinal_accel_mps2"] <= -3.0))

    brake_rate = ratio_over_threshold(df["brake"], 0.5, use_abs=False) if "brake" in df.columns else 0.0
    yaw_risk = ratio_over_threshold(df["yaw_rate_dps"], 25.0, use_abs=True) if "yaw_rate_dps" in df.columns else 0.0
    lateral_risk = ratio_over_threshold(df["lateral_accel_mps2"], 2.5, use_abs=True) if "lateral_accel_mps2" in df.columns else 0.0
    steering_rate_risk = ratio_over_threshold(df["steering_rate_degps"], 80.0, use_abs=True) if "steering_rate_degps" in df.columns else 0.0

    lane_departure_risk = ratio_over_threshold(df["lane_offset_m"], 0.6, use_abs=True) if "lane_offset_m" in df.columns else 0.0
    ttc_risk = float(np.mean(df["ttc_s"] <= 3.0)) if "ttc_s" in df.columns else 0.0
    headway_risk = float(np.mean(df["headway_s"] <= 1.5)) if "headway_s" in df.columns else 0.0

    penalty = (
        15 * harsh_accel
        + 20 * harsh_brake
        + 10 * brake_rate
        + 15 * yaw_risk
        + 10 * lateral_risk
        + 10 * steering_rate_risk
        + 10 * lane_departure_risk
        + 5 * ttc_risk
        + 5 * headway_risk
    )

    score = float(np.clip(100 - penalty, 0, 100))

    details = {
        "harsh_accel_rate": harsh_accel,
        "harsh_brake_rate": harsh_brake,
        "brake_rate": brake_rate,
        "yaw_risk_rate": yaw_risk,
        "lateral_risk_rate": lateral_risk,
        "steering_rate_risk": steering_rate_risk,
        "lane_departure_risk": lane_departure_risk,
        "ttc_risk": ttc_risk,
        "headway_risk": headway_risk,
        "safety_penalty": penalty,
    }

    return score, details


def calculate_smoothness_score(df: pd.DataFrame, signal_summary: pd.DataFrame) -> tuple[float, dict]:
    """
    Smoothness Score.
    Jerk, steering rate, yaw rate, FFT/STFT 이벤트 중심으로 계산한다.
    road_curvature가 있으면 조향 관련 고주파 감점을 완화할 수 있다.
    """
    jerk_rms = float(np.sqrt(np.mean(df["jerk_magnitude_mps3"] ** 2))) if "jerk_magnitude_mps3" in df.columns else 0.0
    steering_rate_rms = float(np.sqrt(np.mean(df["steering_rate_degps"] ** 2))) if "steering_rate_degps" in df.columns else 0.0
    yaw_std = float(df["yaw_rate_dps"].std()) if "yaw_rate_dps" in df.columns else 0.0

    fft_mean = float(signal_summary["fft_high_freq_ratio"].mean()) if len(signal_summary) > 0 else 0.0
    stft_mean = float(signal_summary["stft_event_ratio"].mean()) if len(signal_summary) > 0 else 0.0

    # 도로 곡률 보정
    # road_curvature_1m가 있으면 곡선도로에서 조향 고주파 감점을 완화한다.
    road_context_weight = 1.0
    if "road_curvature_1m" in df.columns:
        curvature = df["road_curvature_1m"].abs()
        curvature_norm = np.clip(curvature / (curvature.quantile(0.95) + 1e-6), 0, 1)
        alpha = 0.4
        road_context_weight = float(1 - alpha * curvature_norm.mean())

    adjusted_fft = fft_mean * road_context_weight
    adjusted_stft = stft_mean * road_context_weight

    jerk_risk = normalize_risk(jerk_rms, 10.0)
    steering_risk = normalize_risk(steering_rate_rms, 100.0)
    yaw_risk = normalize_risk(yaw_std, 30.0)
    fft_risk = normalize_risk(adjusted_fft, 1.0)
    stft_risk = normalize_risk(adjusted_stft, 1.0)

    penalty = (
        25 * jerk_risk
        + 20 * steering_risk
        + 20 * yaw_risk
        + 20 * fft_risk
        + 15 * stft_risk
    )

    score = float(np.clip(100 - penalty, 0, 100))

    details = {
        "jerk_rms": jerk_rms,
        "steering_rate_rms": steering_rate_rms,
        "yaw_std": yaw_std,
        "fft_hfr_mean": fft_mean,
        "stft_event_mean": stft_mean,
        "road_context_weight": road_context_weight,
        "adjusted_fft_hfr": adjusted_fft,
        "adjusted_stft_event": adjusted_stft,
        "smoothness_penalty": penalty,
    }

    return score, details


def calculate_eco_like_score(df: pd.DataFrame) -> tuple[float, dict]:
    """
    CARLA에는 rpm이 없을 수 있으므로 Eco-like Score로 계산한다.
    OBD/CAN 데이터에서 rpm이 있으면 high_rpm_ratio를 추가로 사용할 수 있다.
    """
    throttle_spike = 0.0
    if "throttle" in df.columns:
        throttle_diff = df["throttle"].diff().fillna(0).abs()
        throttle_spike = float(np.mean(throttle_diff >= 0.25))

    brake_usage = ratio_over_threshold(df["brake"], 0.2, use_abs=False) if "brake" in df.columns else 0.0
    harsh_accel = float(np.mean(df["longitudinal_accel_mps2"] >= 2.5)) if "longitudinal_accel_mps2" in df.columns else 0.0
    harsh_brake = float(np.mean(df["longitudinal_accel_mps2"] <= -3.0)) if "longitudinal_accel_mps2" in df.columns else 0.0

    speed_std = float(df["speed_kmh"].std()) if "speed_kmh" in df.columns else 0.0
    speed_instability = normalize_risk(speed_std, 30.0)

    high_rpm_ratio = 0.0
    if "rpm" in df.columns:
        high_rpm_ratio = float(np.mean(df["rpm"] >= 3000))

    penalty = (
        20 * throttle_spike
        + 15 * brake_usage
        + 20 * harsh_accel
        + 20 * harsh_brake
        + 15 * speed_instability
        + 10 * high_rpm_ratio
    )

    score = float(np.clip(100 - penalty, 0, 100))

    details = {
        "throttle_spike_rate": throttle_spike,
        "brake_usage_rate": brake_usage,
        "harsh_accel_rate": harsh_accel,
        "harsh_brake_rate": harsh_brake,
        "speed_std": speed_std,
        "speed_instability": speed_instability,
        "high_rpm_ratio": high_rpm_ratio,
        "eco_like_penalty": penalty,
    }

    return score, details


def calculate_total_score(
    safety: float,
    smoothness: float,
    eco: float,
    safety_details: dict,
    smoothness_details: dict,
    eco_details: dict,
) -> dict:
    """
    Risk 기반 가중치.
    Ws, Wm, We는 각 Risk를 정규화해서 계산한다.
    """
    safety_risk = np.clip(safety_details["safety_penalty"] / 100, 0, 1)
    smoothness_risk = np.clip(smoothness_details["smoothness_penalty"] / 100, 0, 1)
    eco_risk = np.clip(eco_details["eco_like_penalty"] / 100, 0, 1)

    total_risk = safety_risk + smoothness_risk + eco_risk

    if total_risk == 0:
        ws = wm = we = 1 / 3
    else:
        ws = safety_risk / total_risk
        wm = smoothness_risk / total_risk
        we = eco_risk / total_risk

    total_score = ws * safety + wm * smoothness + we * eco

    return {
        "Total Score": float(total_score),
        "Safety Score": float(safety),
        "Smoothness Score": float(smoothness),
        "Eco-like Score": float(eco),
        "Ws": float(ws),
        "Wm": float(wm),
        "We": float(we),
        "Safety Risk": float(safety_risk),
        "Smoothness Risk": float(smoothness_risk),
        "Eco Risk": float(eco_risk),
    }