# Reinforcement Learning based Traffic Light Controller

## Overview
A Reinforcement Learning (RL) based traffic light control system aimed at managing routes with heavy traffic from different directions. It uses real-time monitoring and dynamic adjustment of traffic light timings to minimize waiting times, reduce emissions, and improve traffic flow efficiency. This project is being developed for Smart India Hackathon 2024, under problem statement ID 1607.

## The Reinforcement Learning Formulation

The task of traffic management at road intersections is modelled as a Reinforcement Learning task by depicting traffic conditions as accurately as possible to act as a novel state space, coming up with a novel reward function to act as an effective feedback mechanism for the RL agent, and setting the action space according to the desired level of agent autonomy. For developing the RL agent, the state-of-the-art Soft Actor Critic RL algorithm has been used which incorporates automatic entropy based exploration factor.

### State Space Formulation
The RL agent receives a comprehensive representation of the current traffic scenario, encoded in the following state variables:

- **Total number of vehicles (halted + moving) in each lane**: Vector of dimension (num_vehicle_lanes, ).
- **Number of halted vehicles in each lane**: Vector of dimension (num_vehicle_lanes, ).
- **Sum of waiting times of normal vehicles in each lane**: Vector of dimension (num_vehicle_lanes, ). Waiting time is lane-dependent and defined as the time measured from when a vehicle first stops in that lane until it either leaves that lane (for outgoing lanes) or enters an outgoing lane (for incoming lanes).
- **Sum of waiting times of priority vehicles in each lane**: Vector of dimension (num_vehicle_lanes, ). The same waiting time definition also applies here, ensuring accurate lane-specific measurements.
- **Vehicle position grid**: A binary image/grid representing the positions of vehicles.
- **Vehicle speed grid**: An image/grid representation of vehicle speeds (all elements are non-negative).
- **Vehicle acceleration grid**: An image/grid representation of vehicle accelerations (positive for speed increase, negative for decrease).
- **Priority vehicle indicator grid**: A binary image/grid representing the positions of priority vehicles (e.g., emergency vehicles).
- **Total number of pedestrians (halted + moving) in each lane**: Vector of dimension (num_pedestrian_lanes, ).
- **Number of halted pedestrians in each lane**: Vector of dimension (num_pedestrian_lanes, ).
- **Sum of waiting times of pedestrians in each lane**: Vector of dimension (num_pedestrian_lanes, ). Waiting time is defined similarly for pedestrians, from when they first halt until they leave or enter an outgoing lane.
- **Pedestrian position grid**: A binary image/grid representing the positions of pedestrians.
- **Pedestrian speed grid**: An image/grid representation of pedestrian speeds (all elements are non-negative).
- **Pedestrian acceleration grid**: An image/grid representation of pedestrian accelerations (positive for speed increase, negative for decrease).
- **Current traffic light phase**: Current phase of the traffic light system.
- **Next traffic light phase**: Next planned phase of the traffic light system. (This is part of the state space only for an autonomous level-1 agent. The next phase is part of the action space for an autonomous level-2 agent, and thus is predicted by the agent itself)
- **Time since last phase change**: A single value indicating how long the current phase has been active.
- **Visibility (in kilometers)**: A single value indicating the current visibility conditions. Poor visibility may require longer light phases to give vehicles and pedestrians extra time to react.
- **Daytime indicator**: A binary flag indicating daytime (1 for daytime, 0 for nighttime).

### Reward Function Formulation
The RL agent uses a reward function to guide learning, balancing traffic efficiency and safety. The components of the reward function include:

- **Sudden Phase Change**: Indicates phase changes that occur too quickly, below the minimum duration allowed for that type of phase. Minimum durations are adapted based on visibility conditions to prevent accidents in low visibility. [Goal: Reduce]
- **Slow Phase Change**: Indicates phase changes that occur too slowly, exceeding the maximum duration allowed. Maximum durations are visibility-dependent to ensure traffic remains responsive. [Goal: Reduce]
- **Noise Emission**: Total number of incoming lanes whose noise emission exceeds the permissible limits set by the Central Pollution Control Board (CPCB) of India. The permissible limits are:
  - Industrial areas: 75 dB (daytime), 70 dB (nighttime)
  - Commercial areas: 65 dB (daytime), 55 dB (nighttime)
  - Residential areas: 55 dB (daytime), 45 dB (nighttime)

  Phase durations should be adjusted to minimize noise, especially during nighttime in residential areas. [Goal: Reduce]
- **Environmental Cost**: Sum of CO2, CO, HC, NOx, and PMx emission rates (in mg/s) of entities in all the incoming lanes. [Goal: Reduce]
- **Intermediate Halt**: Indicates scenarios where entities (vehicles or pedestrians) halt while crossing an intersection, as it often indicates a near-collision or inefficient traffic flow. A speed below 0.1 m/s is considered a halt. This is heavily penalized to encourage uninterrupted movement. [Goal: Reduce]
- **Collision**: Number of collisions involving vehicles and/or pedestrians to ensure safety. [Goal: Reduce]
- **Pedestrian Waiting Time**: Sum of waiting times of pedestrians in all the incoming pedestrian lanes. [Goal: Reduce]
- **Priority Vehicle Waiting Time**: Sum of waiting times of priority vehicles in all the incoming vehicle lanes. [Goal: Reduce]
- **Normal Vehicle Waiting Time**: Sum of waiting times of normal vehicles in all the incoming vehicle lanes. [Goal: Reduce]
- **Priority Vehicle Speed**: Average speed of priority vehicles in all the incoming vehicle lanes. [Goal: Increase]
- **Normal Vehicle Speed**: Average speed of normal vehicles in all the incoming vehicle lanes. [Goal: Increase]
- **Pedestrian Speed**: Average speed of pedestrian in all the incoming pedestrian lanes. [Goal: Increase]
- **Pedestrian Queue Length**: Sum of pedestrian queue lengths of each incoming pedestrian lane. [Goal: Reduce]
- **Priority Vehicle Queue Length**: Sum of priority vehicle queue lengths of each incoming vehicle lane. [Goal: Reduce]
- **Normal Vehicle Queue Length**: Sum of normal vehicle queue lengths of each incoming vehicle lane. [Goal: Reduce]

### Action Space Formulation
Modelling of the action space of the RL agent can be done in 3 possible ways which decides the level of autonomy the agent has while managing traffic:
- **Autonomous Level 1**: Here we manually decide the traffic light phases as well as a periodic cycle of phases that is to be followed by the agent. Each traffic light phase is an encoding of the various traffic lights in the road intersection. For example: Hypothetically, if there are only 4 traffic lights in the road intersection, then one of the traffic light phases might be: "red for the first two lights and green for the remaining two lights". At every time instant, the RL agent takes a binary action:
- - **action = 1**: Change the traffic light phase to the pre-defined "next phase" in the phase cycle.
- - **action = 0**: Keep the traffic light phase as it is.
- **Autonomous Level 2**: Here we manually decide the traffic light phases only. At every time instant, the RL agent predicts the next traffic light phase, which can either be same as the current traffic light phase, or a different one.
- **Autonomous Level 3**: This is the highest level of autonomy the RL agent can be given, where there is no manual intervention. The RL agent is free to control each individual "traffic light" separately. Thus, at every time instant, the RL agent predicts an encoding of the traffic lights at the road intersection, which acts as the next traffic light phase -- this can either be same as the current traffic light phase, or a different one.

With increasing level of RL agent autonomy, training becomes significantly challenging -- we need to train for prolonged durations and enforce certain restrictions on the RL agent. 

## Key aspects of this project
- **More than 50% reduction** in traffic waiting times and
environmental cost as compared to traditional pre-set
timing based traffic light control mechanisms.
- **No human intervention needed**: The RL agent changes
the traffic light phase as and when needed.
- **Novel modelling** of weather factors, state space and
reward function – helps train the agent in highly
realistic data
- **Real world integration**: YOLO-v8 and DeepSort models are used to extract real-time real-world traffic data, which is then fed into our pre-trained RL agent to manage traffic.

## Requirements 
### RL Algorithm Training and Evaluation on Simulated Traffic Data:
- **Traffic simulator**: SUMO –
Simulation of Urban MObility
- **Frameworks**: PyTorch,
stable_baselines3 (for RL algorithm),
OpenAI Gym
- **Programming language**: Python

### RL Algorithm Inference/Usage in Real-World Scenario:
- **Hardware Requirements**: Aerial view camera mounted at road intersections, and a computational device with capabilities similar to an average modern laptop.
- **Software Requirements**: 
  - Our Pre-trained Soft Actor Critic RL model for controlling traffic lights.
  - YOLO-v8 and DeepSort for real-time traffic data extraction from a real-world setting.
  - Frameworks -- PyTorch, stable_baselines3 (for RL algorithm) and OpenAI Gym.
  - Programming language -- Python

## Feasibility and Scalability
- **Seamless Integration**: Only an overhead camera and a computational device are needed.
- **Scalability**: Multi-agent RL capabilities allow managing multiple intersections efficiently.
- **Technological Maturity**: The use of well-established algorithms and frameworks supports real-world feasibility.

## Impact and Benefits
- **Social Impact**: Reduction in traffic-related stress, shorter waiting times, and improved quality of life for commuters.
- **Economic Benefits**: Lower fuel consumption, reduced logistics costs, and less vehicle wear and tear.
- **Environmental Benefits**: Decreased emissions of pollutants like CO2 and NOx, contributing to cleaner air in urban areas.
- **Safety**: Improved traffic flow reduces the likelihood of collisions, enhancing road safety.

## Project R&D Status
- A prototype of the autonomous-level-1 SAC RL agent has been developed and trained for 2.5 days.
- A prototype of the autonomous-level-2 SAC RL agent has been developed and is currently being trained.
- Work on the development of the autonomous-level-3 SAC RL agent is yet to begin and would commence after the successful training of the autonomous-level-2 agent.
- Real-world integration using YOLO and DeepSort models is 70% complete.
- Future improvements include, training on simulated scenarios involving accidents and road closures.

## References
1. [IntelliLight: A Reinforcement Learning Approach for Intelligent Traffic Light Control](https://dl.acm.org/doi/10.1145/3219819.3220096)
2. [SUMO Documentation](https://sumo.dlr.de/docs/index.html)
3. [YOLO Ultralytics](https://docs.ultralytics.com/modes/track/)
