# AI Based Traffic Light Controller

## Description
To tackle the traffic issues in urban areas, we have proposed a smart **Reinforcement Learning(RL)** based traffic light controller capable of **real-time monitoring** of traffic conditions and **adaptation of traffic light timings** accordingly. 

## Algorithm
We have used Soft Actor Critic algorithm for developing the RL agent which involves training for multiple episodes with a novel state space and reward function. Extensive research has been conducted to develop state spaces and reward functions for designing a reinforcement learning-based agent that not only ensures a smooth traffic experience but also promotes eco-friendly practices. For training purposes, SUMO(Simulation Of Urban MObility), a traffic simulator has been used.

### State Space
- Number of vehicles in lanes
- Vehicle position, speed and acceleration values
- Waiting time of vehicles
- Priority vehicle indicator grid
- Similar factors for pedestrians
- Current traffic light phase
- Upcoming traffic light phase
- Time since last phase change
- Visibiliy (in km)
- Daytime

### Reward Function
Weighted sum of following factors has been considered as reward function
- Queue Length
- Speed of vehicles
- Waiting time of vehicles
- Priority vehicles
- Pedestrians
- Sum of CO2, CO, HC, NOx and PMx emission rates
- Noise emission
- Collision indicator
- Sudden or slow phase changes

The state-space and reward-function have elements like **priority grid** to support emergency vehicles and **Pollutants emission rates** to reduce pollution.



