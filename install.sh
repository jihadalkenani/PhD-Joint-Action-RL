#!/bin/bash

# Check if ROS is installed
if ! command -v roscore &> /dev/null; then
    echo "ROS is not installed. Please install ROS first."
    echo "Visit http://wiki.ros.org/ROS/Installation for instructions."
    exit 1
fi

# Install Python dependencies
pip install tensorflow numpy matplotlib pandas seaborn

# Create a ROS workspace if it doesn't exist
if [ ! -d ~/catkin_ws ]; then
    mkdir -p ~/catkin_ws/src
    cd ~/catkin_ws
    catkin_make
    source devel/setup.bash
fi

# Copy this package to the workspace
cp -r $(dirname "$0") ~/catkin_ws/src/joint_action_rl

# Build the package
cd ~/catkin_ws
catkin_make

echo "Installation complete. You can now use the joint_action_rl package."
echo "Don't forget to source the workspace: source ~/catkin_ws/devel/setup.bash"
