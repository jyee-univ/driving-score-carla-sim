# 수집 코드 맨 위에 추가
import math
import carla


def angle_diff(a, b):
    diff = a - b

    while diff > math.pi:
        diff -= 2 * math.pi

    while diff < -math.pi:
        diff += 2 * math.pi

    return diff


def distance_2d(loc1, loc2):
    dx = loc2.x - loc1.x
    dy = loc2.y - loc1.y

    return math.sqrt(dx * dx + dy * dy)


def get_avg_road_curvature(carla_map, vehicle, step=2.0, n=5):
    """
    CARLA waypoint 기반 도로 곡률 계산
    return: road_curvature_1m
    """

    wp = carla_map.get_waypoint(
        vehicle.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )

    if wp is None:
        return 0.0

    curvatures = []

    for _ in range(n):
        next_wps = wp.next(step)

        if not next_wps:
            break

        wp_next = next_wps[0]

        yaw1 = math.radians(wp.transform.rotation.yaw)
        yaw2 = math.radians(wp_next.transform.rotation.yaw)

        ds = distance_2d(
            wp.transform.location,
            wp_next.transform.location
        )

        if ds > 0.001:
            curvature = abs(angle_diff(yaw2, yaw1)) / ds
            curvatures.append(curvature)

        wp = wp_next

    if len(curvatures) == 0:
        return 0.0

    return sum(curvatures) / len(curvatures)
