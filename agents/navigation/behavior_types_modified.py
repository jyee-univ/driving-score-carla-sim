class Cautious(object):
    """Class for Cautious agent."""
    max_speed = 35
    speed_lim_dist = 8
    speed_decrease = 15
    safety_time = 4
    min_proximity_threshold = 15
    braking_distance = 8
    tailgate_counter = 0


class Normal(object):
    """Class for Normal agent."""
    max_speed = 50
    speed_lim_dist = 3
    speed_decrease = 10
    safety_time = 3
    min_proximity_threshold = 10
    braking_distance = 5
    tailgate_counter = 0


class Aggressive(object):
    """Class for Aggressive agent."""
    max_speed = 70
    speed_lim_dist = 2
    speed_decrease = 7
    safety_time = 2
    min_proximity_threshold = 7
    braking_distance = 4
    tailgate_counter = -1
