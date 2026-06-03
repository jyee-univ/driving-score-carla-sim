from collect_carla_signals import run_scenario


if __name__ == "__main__":
    run_scenario(
        mode="aggressive",
        output_csv="data/sample/carla_aggressive_signals.csv",
        duration_s=60.0,
        target_speed=55.0,
    )