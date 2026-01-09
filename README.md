# Semantic-Aware-Navigation

This project implements a semantic-aware navigation system in ROS 2.  
Instead of treating the environment as just obstacles and free space, the robot also understands *what* it is seeing (for example walls, floor, objects) and uses that information while navigating.

The system combines:
- Gazebo simulation
- Nav2 for navigation
- A semantic segmentation node
- Projection of semantic labels into the navigation costmap

This allows the robot to avoid certain types of regions (like obstacles or restricted zones) more intelligently than standard navigation.

## What it contains
- A custom Gazebo world and robot model  
- ROS 2 nodes for semantic segmentation and visualization  
- A semantic costmap layer that affects path planning  
- A launch file to run the full pipeline  

---- in progress :) ----
