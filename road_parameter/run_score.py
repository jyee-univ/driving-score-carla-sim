import pandas as pd
from scoring_with_curvature import compute_driver_scores


df = pd.read_csv("carla_signals_with_curvature.csv")

scores = compute_driver_scores(df)

for key, value in scores.items():
    print(f"{key}: {value}")
