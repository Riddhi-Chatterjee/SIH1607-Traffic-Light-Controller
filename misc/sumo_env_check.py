import sumolib
import traci
import time
import numpy as np

import xml.etree.ElementTree as ET

def add_emergency_vehicle_type_to_route(input_route_file, output_route_file=None):
    """
    Add an emergency vehicle type to the vTypeDistribution block in a SUMO .rou.xml file.
    
    Parameters:
    input_route_file (str): Path to the input SUMO route file (.rou.xml).
    output_route_file (str): Path to save the modified SUMO route file. If None, the original file is overwritten.
    """
    
    # Parse the XML file
    tree = ET.parse(input_route_file)
    root = tree.getroot()

    # Find the <vTypeDistribution> block
    vtype_distribution = root.find(".//vTypeDistribution")

    if vtype_distribution is None:
        print("vTypeDistribution block not found.")
        return

    # Check if the emergency vType already exists
    existing_vtype = vtype_distribution.find(".//vType[@id='emergency']")
    
    if existing_vtype is None:
        # If not found, create a new vType for emergency vehicles
        emergency_vtype = ET.Element("vType")
        emergency_vtype.set("id", "emergency")
        emergency_vtype.set("length", "5.0")
        emergency_vtype.set("minGap", "2.0")
        emergency_vtype.set("guiShape", "emergency")
        emergency_vtype.set("color", "0,0,1")  # Blue color
        emergency_vtype.set("probability", "1")  # Set probability for emergency vehicles

        # Add the new vType to the vTypeDistribution block
        vtype_distribution.append(emergency_vtype)
        print("Added emergency vehicle type to vTypeDistribution.")
    else:
        # Update the existing emergency vType
        existing_vtype.set("length", "5.0")
        existing_vtype.set("minGap", "2.0")
        existing_vtype.set("guiShape", "emergency")
        existing_vtype.set("color", "0,0,1")  # Blue color
        existing_vtype.set("probability", "1")
        print("Updated existing emergency vehicle type in vTypeDistribution.")
    
    # If no output file is specified, overwrite the original file
    if output_route_file is None:
        output_route_file = input_route_file

    # Write the modified XML to the output file
    tree.write(output_route_file, encoding="UTF-8", xml_declaration=True)
    print(f"Route file saved to: {output_route_file}")


input_route_file = "/opt/homebrew/Cellar/sumo/1.20.0/share/sumo/tools/game/fkk_in/fkk_in.rou.xml"
output_route_file = None  # If None, the original file will be overwritten

add_emergency_vehicle_type_to_route(input_route_file, output_route_file)



def set_traffic_light_logic(current_phase):
    traffic_light_ids = traci.trafficlight.getIDList()
    
    phases=[]
    phases.append(traci.trafficlight.Phase(duration=10000.0, state='rrGGGGrrrrrrrrrrrr', minDur=10000.0, maxDur=10000.0, name='1'))
    phases.append(traci.trafficlight.Phase(duration=10000.0, state='rrGyyyrrrrrrrrrrrr', minDur=10000.0, maxDur=10000.0, name='2'))
    phases.append(traci.trafficlight.Phase(duration=10000.0, state='rrGrrrrrrrrrrrrrrr', minDur=10000.0, maxDur=10000.0, name='3'))
    phases.append(traci.trafficlight.Phase(duration=10000.0, state='GGGGrrrrrrrrrrrrrr', minDur=10000.0, maxDur=10000.0, name='4'))
    phases.append(traci.trafficlight.Phase(duration=10000.0, state='yyGyrrrrrrrrrrrrrr', minDur=10000.0, maxDur=10000.0, name='5'))
    phases.append(traci.trafficlight.Phase(duration=10000.0, state='rrGrrrrrrrrrrrrrrr', minDur=10000.0, maxDur=10000.0, name='6'))
    phases.append(traci.trafficlight.Phase(duration=10000.0, state='rrGGrrrrrrrGGGrrGr', minDur=10000.0, maxDur=10000.0, name='7'))
    phases.append(traci.trafficlight.Phase(duration=10000.0, state='rrGyrrrrrrryGyrrGr', minDur=10000.0, maxDur=10000.0, name='8'))
    phases.append(traci.trafficlight.Phase(duration=10000.0, state='rrGrrrrrrrrrrrrrrr', minDur=10000.0, maxDur=10000.0, name='9'))
    phases.append(traci.trafficlight.Phase(duration=10000.0, state='rrGrrrGGGrrrrrGrrG', minDur=10000.0, maxDur=10000.0, name='10'))
    phases.append(traci.trafficlight.Phase(duration=10000.0, state='rrGrrryyyrrrrrGrry', minDur=10000.0, maxDur=10000.0, name='11'))
    phases.append(traci.trafficlight.Phase(duration=10000.0, state='rrGrrrrrrrrrrrrrrr', minDur=10000.0, maxDur=10000.0, name='12'))
    phases.append(traci.trafficlight.Phase(duration=10000.0, state='rrGrGrrrGGGrrrrGrr', minDur=10000.0, maxDur=10000.0, name='13'))
    phases.append(traci.trafficlight.Phase(duration=10000.0, state='rrGryrrryyGrrrryrr', minDur=10000.0, maxDur=10000.0, name='14'))
    phases.append(traci.trafficlight.Phase(duration=10000.0, state='rrGrrrrrrrrrrrrrrr', minDur=10000.0, maxDur=10000.0, name='15'))
    
    new_program = traci.trafficlight.Logic(programID='Pnew', type=0, currentPhaseIndex=current_phase, phases=phases, subParameter={})

    traci.trafficlight.setProgramLogic(traffic_light_ids[1], new_program)

num_eval_episodes = 1
num_eval_episode_steps = 5000

# Load the network file
net_file = "/opt/homebrew/Cellar/sumo/1.20.0/share/sumo/tools/game/fkk_in/ingolstadt.net.xml.gz"
net = sumolib.net.readNet(net_file)

# Get the network's boundaries
bounding_box = net.getBoundary()

# Calculate the width and height of the area
width = bounding_box[2] - bounding_box[0]
height = bounding_box[3] - bounding_box[1]

AREA_SIZE = (width, height)  # Width and height of the area in meters

print("AREA_SIZE:", AREA_SIZE)
print("LEFTMOST:", bounding_box[0])
print("RIGHTMOST:", bounding_box[2])
print("BOTTOMMOST:", bounding_box[1])
print("TOPMOST:", bounding_box[3])

GRID_SIZE = (int(AREA_SIZE[0]), int(AREA_SIZE[1]))  # Rows and columns in the grid 
GRID_RES = (AREA_SIZE[0] / GRID_SIZE[0], AREA_SIZE[1] / GRID_SIZE[1])    

# Path to SUMO binary
sumoBinary = "/opt/homebrew/bin/sumo-gui"
sumoCmd = [sumoBinary, "-c", "/opt/homebrew/Cellar/sumo/1.20.0/share/sumo/tools/game/fkk_in.sumocfg"]

for episode in range(num_eval_episodes):
    
    # Start the SUMO simulation with TraCI
    traci.start(sumoCmd)
    
    # print(traci.junction.getShape("gneJ21"))
    
    set_traffic_light_logic(1)
    
    simulation_time_at_phase_change = traci.simulation.getTime()

    # Loop to interact with the SUMO simulation
    episode_steps = 0
    while episode_steps < num_eval_episode_steps:
        
        lanes = traci.lane.getIDList()
        queue_length = {lane: traci.lane.getLastStepHaltingNumber(lane) for lane in lanes}
        vehicle_count = {lane: traci.lane.getLastStepVehicleNumber(lane) for lane in lanes}
        avg_waiting_time = {lane: traci.lane.getWaitingTime(lane) for lane in lanes}
        traffic_light_ids = traci.trafficlight.getIDList()
        current_phase = {tl: traci.trafficlight.getPhase(tl) for tl in traffic_light_ids}
        # print("CURRENT_PHASE: "+str(current_phase))
        num_collided_vehicles = traci.simulation.getCollidingVehiclesNumber()
        # print(num_collided_vehicles)
        # print(traci.simulation.getDeltaT())
        
        current_time = traci.simulation.getTime()
        
        traffic_light_ids = traci.trafficlight.getIDList()
        action = 0
        if (current_time - simulation_time_at_phase_change) >= 20:
            action = 1
        
        # Advance simulation by one step
        print(traci.simulation.getCurrentTime()/1000)
        traci.trafficlight.setPhase(traffic_light_ids[1], traci.trafficlight.getPhase(traffic_light_ids[1]))
        traci.simulationStep()
        traci.trafficlight.setPhase(traffic_light_ids[1], traci.trafficlight.getPhase(traffic_light_ids[1]))

        # Set traffic light phase according to action chosen
        if action == 1:
            traci.trafficlight.setPhase(traffic_light_ids[1], (traci.trafficlight.getPhase(traffic_light_ids[1]) + 1) % 15)
            simulation_time_at_phase_change = traci.simulation.getTime()
        else:
            traci.trafficlight.setPhase(traffic_light_ids[1], traci.trafficlight.getPhase(traffic_light_ids[1]))
        
        # Initialize the grid with zeros
        vehicle_grid = np.zeros((GRID_SIZE[1], GRID_SIZE[0]))

        # Get all vehicle IDs
        vehicle_ids = traci.vehicle.getIDList()

        # Loop through all vehicles and place them on the grid
        for veh_id in vehicle_ids:
            # print(traci.vehicle.getSpeed(veh_id))
            # print(traci.vehicle.getAcceleration(veh_id))
            
            # print(traci.vehicle.getTypeID(veh_id))
            # print(traci.vehicle.getTypeID(veh_id) == "passenger")
            # traci.vehicle.setType(veh_id, "emergency")
            # Get the position of the vehicle
            x, y = traci.vehicle.getPosition(veh_id)
            
            # Map the position to the grid indices
            row = int(y // GRID_RES[1])  # Rows are indexed by the y-coordinate
            col = int(x // GRID_RES[0])  # Columns are indexed by the x-coordinate
            
            # Ensure the indices are within the grid bounds
            if 0 <= row < GRID_SIZE[1] and 0 <= col < GRID_SIZE[0]:
                vehicle_grid[row, col] += 1  # Increment the count at the grid position
        print("")


        # for row in vehicle_grid[185:205, 80:200]:
        #     for col in [int(x) for x in row]:
        #         print(col, end="")
        #     print("")
        # print("\n")
        
        episode_steps += 1
        
    # End SUMO simulation
    traci.close()