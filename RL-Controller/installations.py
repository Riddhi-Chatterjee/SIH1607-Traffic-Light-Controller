import os
os.system("sudo add-apt-repository -y ppa:sumo/stable")
os.system("sudo apt-get update -y")
os.system("sudo apt-get install -y sumo sumo-tools sumo-doc")
os.system("pip install sumolib traci")
os.system("pip install torchviz")
os.system("pip install stable-baselines3 gym")