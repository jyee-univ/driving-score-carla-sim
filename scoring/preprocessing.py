import pandas as pd
import numpy as np


def load_carla_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if "timestamp_s" not in df.columns:
        raise ValueError("timestamp_s column is required.")

    df = df.sort_values("timestamp_s").reset_index(drop=True)
    return df


def remove_initial_unstable_data(df: pd.DataFrame, cut_seconds: float = 2.0) -> pd.DataFrame:
    """
    CARLA 차량 spawn 직후에는 acceleration, jerk가 비정상적으로 튈 수 있으므로
    초기 몇 초 데이터를 제거한다.
    """
    start_time = df["timestamp_s"].min()
    df = df[df["timestamp_s"] >= start_time + cut_seconds].copy()
    return df.reset_index(drop=True)


def clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    acceleration, gyro, jerk 계열의 극단적인 이상치를 percentile 기준으로 clipping.
    """
    clip_cols = [
        "accel_x_mps2",
        "accel_y_mps2",
        "accel_z_mps2",
        "gyro_x_radps",
        "gyro_y_radps",
        "gyro_z_radps",
        "steering_rate_degps",
        "jerk_x_mps3",
        "jerk_y_mps3",
        "jerk_z_mps3",
        "jerk_magnitude_mps3",
    ]

    for col in clip_cols:
        if col in df.columns:
            low = df[col].quantile(0.01)
            high = df[col].quantile(0.99)
            df[col] = df[col].clip(low, high)

    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "gyro_z_radps" in df.columns:
        df["yaw_rate_dps"] = np.rad2deg(df["gyro_z_radps"])

    if "accel_y_mps2" in df.columns:
        df["lateral_accel_mps2"] = df["accel_y_mps2"]

    if "accel_x_mps2" in df.columns:
        df["longitudinal_accel_mps2"] = df["accel_x_mps2"]

    return df


def preprocess_carla_data(path: str) -> pd.DataFrame:
    df = load_carla_csv(path)
    df = remove_initial_unstable_data(df, cut_seconds=2.0)
    df = clip_outliers(df)
    df = add_derived_columns(df)
    return df