#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import numpy as np

class VisualizationNode(Node):
    """
    Node for visualizing semantic navigation data in RViz.
    Creates overlays, markers, and debug visualizations.
    """
    
    def __init__(self):
        super().__init__('visualization_node')
        
        # Parameters
        self.declare_parameter('semantic_topic', '/semantic/segmentation')
        self.declare_parameter('costmap_topic', '/semantic/costmap')
        
        semantic_topic = self.get_parameter('semantic_topic').value
        costmap_topic = self.get_parameter('costmap_topic').value
        
        # Bridge
        self.bridge = CvBridge()
        
        # Subscribers
        self.seg_sub = self.create_subscription(
            Image,
            semantic_topic,
            self.segmentation_callback,
            10
        )
        
        self.costmap_sub = self.create_subscription(
            OccupancyGrid,
            costmap_topic,
            self.costmap_callback,
            10
        )
        
        # Publishers
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/semantic/markers',
            10
        )
        
        self.overlay_pub = self.create_publisher(
            Image,
            '/semantic/overlay',
            10
        )
        
        # State
        self.latest_seg = None
        self.latest_costmap = None
        
        # Timer for periodic visualization updates
        self.create_timer(0.5, self.publish_markers)
        
        self.get_logger().info('Visualization Node initialized')
    
    def segmentation_callback(self, msg):
        """Store latest segmentation"""
        try:
            self.latest_seg = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f'Error converting segmentation: {e}')
    
    def costmap_callback(self, msg):
        """Store latest semantic costmap"""
        self.latest_costmap = msg
    
    def publish_markers(self):
        """Publish visualization markers"""
        if self.latest_seg is None:
            return
        
        marker_array = MarkerArray()
        
        # Create marker for each semantic class
        class_info = {
            1: ('Person', [1.0, 0.0, 0.0, 0.8]),      # Red
            2: ('Vehicle', [0.0, 0.0, 1.0, 0.8]),     # Blue
            3: ('Obstacle', [1.0, 0.5, 0.0, 0.8]),    # Orange
        }
        
        for class_id, (name, color) in class_info.items():
            marker = Marker()
            marker.header.frame_id = 'base_footprint'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = f'semantic_{name.lower()}'
            marker.id = class_id
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            
            # Count pixels
            count = np.sum(self.latest_seg == class_id)
            percentage = (count / self.latest_seg.size) * 100
            
            marker.pose.position.x = 0.0
            marker.pose.position.y = class_id * 0.5
            marker.pose.position.z = 1.5
            
            marker.scale.z = 0.2
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = color[3]
            
            marker.text = f'{name}: {percentage:.1f}%'
            marker.lifetime.sec = 1
            
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)
    
    def create_costmap_overlay(self):
        """Create visualization overlay of semantic costmap"""
        if self.latest_costmap is None:
            return None
        
        # Convert costmap to image
        width = self.latest_costmap.info.width
        height = self.latest_costmap.info.height
        data = np.array(self.latest_costmap.data).reshape((height, width))
        
        # Create colored visualization
        colored = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Color mapping
        colored[data == 0] = [255, 255, 255]      # Free - white
        colored[(data > 0) & (data < 100)] = [200, 200, 255]  # Low cost - light blue
        colored[(data >= 100) & (data < 200)] = [255, 200, 100]  # Medium - orange
        colored[data >= 200] = [255, 100, 100]   # High cost - red
        colored[data == 255] = [128, 128, 128]   # Unknown - gray
        
        return colored

def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()