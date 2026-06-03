import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# 현재 파일 기준으로 scoring 폴더를 import 가능하게 설정
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

from preprocessing import preprocess_carla_data
from signal_processing import extract_signal_features
from scoring_function import (
    calculate_safety_score,
    calculate_smoothness_score,
    calculate_eco_like_score,
    calculate_total_score,
)


def run_score(input_path: str, output_dir: str = "data/processed") -> None:
    os.makedirs(output_dir, exist_ok=True)

    print("[1] Loading and preprocessing data...")
    df = preprocess_carla_data(input_path)

    processed_path = os.path.join(output_dir, "carla_signals_preprocessed.csv")
    df.to_csv(processed_path, index=False, encoding="utf-8-sig")
    print(f"Saved preprocessed data: {processed_path}")

    print("[2] Extracting FFT/STFT features...")

    feature_cols = [
        "speed_kmh",
        "throttle",
        "brake",
        "steering_angle_deg",
        "steering_rate_degps",
        "longitudinal_accel_mps2",
        "lateral_accel_mps2",
        "yaw_rate_dps",
        "jerk_magnitude_mps3",
    ]

    # 추후 CARLA에서 추가될 feature
    optional_cols = [
        "lane_offset_m",
        "front_vehicle_distance_m",
        "headway_s",
        "ttc_s",
        "road_curvature_1m",
    ]

    feature_cols.extend([c for c in optional_cols if c in df.columns])

    signal_summary = extract_signal_features(df, feature_cols)

    signal_summary_path = os.path.join(output_dir, "carla_signal_feature_summary.csv")
    signal_summary.to_csv(signal_summary_path, index=False, encoding="utf-8-sig")
    print(f"Saved signal summary: {signal_summary_path}")

    print("[3] Calculating scores...")

    safety_score, safety_details = calculate_safety_score(df)
    smoothness_score, smoothness_details = calculate_smoothness_score(df, signal_summary)
    eco_score, eco_details = calculate_eco_like_score(df)

    total = calculate_total_score(
        safety=safety_score,
        smoothness=smoothness_score,
        eco=eco_score,
        safety_details=safety_details,
        smoothness_details=smoothness_details,
        eco_details=eco_details,
    )

    score_df = pd.DataFrame([total])
    score_path = os.path.join(output_dir, "carla_score_result.csv")
    score_df.to_csv(score_path, index=False, encoding="utf-8-sig")
    print(f"Saved score result: {score_path}")

    detail_df = pd.DataFrame([{**safety_details, **smoothness_details, **eco_details}])
    detail_path = os.path.join(output_dir, "carla_score_detail.csv")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"Saved score details: {detail_path}")

    print("\n===== Driving Score Result =====")
    for key, value in total.items():
        print(f"{key}: {value:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/sample/carla_signals.csv",
        help="Input CARLA CSV path",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory",
    )

    args = parser.parse_args()
    run_score(args.input, args.output_dir)


if __name__ == "__main__":
    main()