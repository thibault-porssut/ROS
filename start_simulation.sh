#!/bin/bash

# This script automates the launch process for the Panda pick-and-place simulation.
# It opens new terminal tabs for each required component in the correct order.

# --- Configuration ---
# The absolute path to your Catkin Workspace
CATKIN_WS_PATH="/home/ubuntu_hi_paris/ws_moveit"

# The absolute path to your Python scripts
SCRIPTS_PATH="/home/ubuntu_hi_paris/Documents/VLA/ROS"

# The name of your main ROS launch file
LAUNCH_PACKAGE="panda_moveit_config"
LAUNCH_FILE="demo_gazebo.launch" # IMPORTANT: Change this to your actual launch file name

echo "--- Starting Panda Pick-and-Place Simulation ---"

# --- Step 1: Launch Gazebo, MoveIt!, and the Link Attacher ---
# This opens a new terminal tab and runs the main roslaunch command.
# The --tab argument opens a new tab in the existing terminal window.
echo "[1/3] Launching Gazebo and MoveIt!..."
gnome-terminal --tab --title="Gazebo & MoveIt!" --command="bash -c 'source ${CATKIN_WS_PATH}/devel/setup.bash; roslaunch ${LAUNCH_PACKAGE} ${LAUNCH_FILE}; exec bash'"

# Give Gazebo and all ROS services time to start up completely.
# This delay is crucial for stability.
echo "Waiting 15 seconds for Gazebo and ROS services to initialize..."
sleep 15

# --- Step 2: Launch the ROS Server ---
# This opens a second tab for the Python 2.7 server that controls MoveIt!.
echo "[2/3] Launching the ROS Server..."
gnome-terminal --tab --title="ROS Server (Py2.7)" --command="bash -c 'source ${CATKIN_WS_PATH}/devel/setup.bash; cd ${SCRIPTS_PATH}; python ros_server.py; exec bash'"

# Give the server a moment to start.
echo "Waiting 5 seconds for the server to start..."
sleep 5

# --- Step 3: Launch the Grasp Manager---
# This opens a third tab for the node that handles the physics attachment in Gazebo.
echo "[3/3] Launching the Grasp Manager..."
gnome-terminal --tab --title="Grasp Manager (Py2.7)" --command="bash -c 'source ${CATKIN_WS_PATH}/devel/setup.bash; cd ${SCRIPTS_PATH}; python grasp_manager.py; exec bash'"

echo "--- All components launched successfully! ---"
echo "You can now run your 'record_expert.py' client in a new terminal."