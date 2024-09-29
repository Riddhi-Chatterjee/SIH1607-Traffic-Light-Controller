# Reinforcement Learning based Traffic Light Controller

## Overview
A Reinforcement Learning (RL) based traffic light control system aimed at managing routes with heavy traffic from different directions. It uses real-time monitoring and dynamic adjustment of traffic light timings to minimize waiting times, reduce emissions, and improve traffic flow efficiency. This project is being developed for Smart India Hackathon 2024, under problem statement ID 1607.

## The Reinforcement Learning Formulation

The task of traffic management at road intersections is modelled as a Reinforcement Learning task by depicting traffic conditions as accurately as possible to act as a novel state space, coming up with a novel reward function to act as an effective feedback mechanism for the RL agent, and setting the action space according to the desired level of agent autonomy.

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

## Key Features
- **Novel State Space and Reward Formulation**: The system uses a unique state representation that captures traffic dynamics, including vehicle count, speed, acceleration, waiting time, priority vehicles, and pedestrians.
- **Soft Actor Critic RL Algorithm**: The SAC algorithm ensures efficient exploration by incorporating entropy-based optimization, allowing the agent to adapt dynamically.
- **Real-Time Traffic Monitoring**: Factors such as time of day, weather, harmful emissions, and collision events are also integrated to make accurate decisions.

## Innovation and Uniqueness
- **50% Reduction in Waiting Times**: The RL-based system demonstrates a significant reduction in vehicle waiting times and environmental costs compared to traditional, preset traffic light systems.
- **No Human Intervention**: The RL agent operates autonomously, making real-time decisions to change traffic light phases as necessary.
- **Scalable and Technologically Mature**: The approach can easily be extended to multiple intersections through Multi-Agent RL, leveraging established frameworks like SUMO, PyTorch, YOLO, and DeepSort.

## Technical Approach
### RL Training
- **Simulator**: SUMO (Simulation of Urban MObility)
- **Frameworks and Tools**: 
  - PyTorch, stable_baselines3 (for RL algorithm), OpenAI Gym for training
  - Python programming language

### Real-World Integration
- **Hardware Requirements**: Aerial view camera mounted at road intersections, and a computational device with capabilities similar to an average modern laptop.
- **Software Requirements**: 
  - Pre-trained SAC RL model for controlling traffic lights.
  - YOLO-v8 and DeepSort for real-time traffic data extraction.
  - PyTorch and stable_baselines3 for inference and adaptation.

### Development Status
- The RL algorithm has been prototyped and tested on simulated traffic data using SUMO.
- Real-world integration with YOLO and DeepSort models is 70% complete.

## Feasibility and Scalability
- **Seamless Integration**: Only an overhead camera and a computational device are needed.
- **Scalability**: Multi-agent RL capabilities allow managing multiple intersections efficiently.
- **Technological Maturity**: The use of well-established algorithms and frameworks supports real-world feasibility.

## Challenges and Strategies
- **Testing and Fine-Tuning**: The prototype is trained on 10+ hours of simulated traffic data but requires more extensive testing before full deployment.
- **Handling Sudden Traffic Events**: Future improvements include training on simulated scenarios involving accidents and road closures.

## Impact and Benefits
- **Social Impact**: Reduction in traffic-related stress, shorter waiting times, and improved quality of life for commuters.
- **Economic Benefits**: Lower fuel consumption, reduced logistics costs, and less vehicle wear and tear.
- **Environmental Benefits**: Decreased emissions of pollutants like CO2 and NOx, contributing to cleaner air in urban areas.
- **Safety**: Improved traffic flow reduces the likelihood of collisions, enhancing road safety.

## References
1. [IntelliLight: A Reinforcement Learning Approach for Intelligent Traffic Light Control](https://dl.acm.org/doi/10.1145/3219819.3220096)
2. [SUMO Documentation](https://sumo.dlr.de/docs/index.html)
3. [YOLO Ultralytics](https://docs.ultralytics.com/modes/track/)
