# 1) cautious
from agents.navigation.behavior_agent import BehaviorAgent

# 안정적이고 조심스러운 운전자
agent = BehaviorAgent(vehicle, behavior='cautious')

destination = random.choice(spawn_points).location
agent.set_destination(destination)

while True:
  world.tick()

  control = agent.run_step()
  vehicle.apply_control(control)

  if agent.done():
    destination = random.choice(spawn_points).locㄴation
    agent.set_destination(destination)

# 2) normal
from agents.navigation.behavior_agent import BehaviorAgent

# 일반적인 운전자
agent = BehaviorAgent(vehicle, behavior='normal')

destination = random.choice(spawn_points).location
agent.set_destination(destination)

while True:
  world.tick()

  control = agent.run_step()
  vehicle.apply_control(control)

  if agent.done():
    destination = random.choice(spawn_points).location
    agent.set_destination(destination)

# 3) aggresive
from agents.navigation.behavior_agent import BehaviorAgent

# 공격적인 운전자
agent = BehaviorAgent(vehicle, behavior='aggressive')

destination = random.choice(spawn_points).location
agent.set_destination(destination)

while True:
  world.tick()

  control = agent.run_step()
  vehicle.apply_control(control)

  if agent.done():
    destination = random.choice(spawn_points).location
    agent.set_destination(destination)

# 만약 CARLA 공식 automatic_control.py를 그대로 쓴다면, 명령어만 다르게 실행하면 구현 가능

# 1) cautious 
python automatic_control.py --agent Behavior --behavior cautious

# 2) normal
python automatic_control.py --agent Behavior --behavior normal

# 3) aggresive
python automatic_control.py --agent Behavior --behavior aggressive
