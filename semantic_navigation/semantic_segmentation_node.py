#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from torchvision import models, transforms

class SemanticSegmentationNode(Node):
    """
    Node that performs semantic segmentation on camera images.
    Uses DeepLabV3 pretrained on COCO/Cityscapes for scene understanding.
    """
    
    def __init__(self):
        super().__init__('semantic_segmentation_node')
        
        # Parameters
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('output_topic', '/semantic/segmentation')
        self.declare_parameter('use_gpu', True)
        self.declare_parameter('model_name', 'deeplabv3_resnet50')
        
        camera_topic = self.get_parameter('camera_topic').value
        output_topic = self.get_parameter('output_topic').value
        use_gpu = self.get_parameter('use_gpu').value
        model_name = self.get_parameter('model_name').value
        
        # Device setup
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')
        
        # Load model
        self.get_logger().info(f'Loading model: {model_name}')
        if model_name == 'deeplabv3_resnet50':
            self.model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        elif model_name == 'deeplabv3_resnet101':
            self.model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        else:
            self.get_logger().error(f'Unknown model: {model_name}')
            return
            
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # COCO/Pascal VOC class names (21 classes)
        self.class_names = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
            'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
            'sofa', 'train', 'tvmonitor'
        ]
        
        # Semantic mapping to navigation-relevant classes
        self.semantic_map = {
            0: 'background',     # Unknown/background
            1: 'vehicle',        # aeroplane -> vehicle
            2: 'vehicle',        # bicycle
            3: 'obstacle',       # bird
            4: 'vehicle',        # boat
            5: 'obstacle',       # bottle
            6: 'vehicle',        # bus
            7: 'vehicle',        # car
            8: 'obstacle',       # cat
            9: 'obstacle',       # chair
            10: 'obstacle',      # cow
            11: 'obstacle',      # table
            12: 'obstacle',      # dog
            13: 'obstacle',      # horse
            14: 'vehicle',       # motorbike
            15: 'person',        # person - most important!
            16: 'obstacle',      # plant
            17: 'obstacle',      # sheep
            18: 'obstacle',      # sofa
            19: 'vehicle',       # train
            20: 'obstacle',      # tv
        }
        
        # Bridge
        self.bridge = CvBridge()
        
        # Subscribers and Publishers
        self.image_sub = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            10
        )
        
        self.seg_pub = self.create_publisher(Image, output_topic, 10)
        self.colored_pub = self.create_publisher(Image, f'{output_topic}/colored', 10)
        
        # Stats
        self.frame_count = 0
        self.get_logger().info('Semantic Segmentation Node initialized')
    
    def image_callback(self, msg):
        """Process incoming camera image"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            
            # Run segmentation
            seg_mask = self.segment_image(cv_image)
            
            # Publish segmentation mask (grayscale with class IDs)
            seg_msg = self.bridge.cv2_to_imgmsg(seg_mask.astype(np.uint8), encoding='mono8')
            seg_msg.header = msg.header
            self.seg_pub.publish(seg_msg)
            
            # Create colored visualization
            colored = self.colorize_segmentation(seg_mask)
            colored_msg = self.bridge.cv2_to_imgmsg(colored, encoding='rgb8')
            colored_msg.header = msg.header
            self.colored_pub.publish(colored_msg)
            
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                self.get_logger().info(f'Processed {self.frame_count} frames')
                
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
    
    def segment_image(self, image):
        """Run semantic segmentation on image"""
        # Preprocess
        input_tensor = self.preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        
        # Get predictions
        output_predictions = output.argmax(0).cpu().numpy()
        
        # Map to simplified semantic classes
        semantic_mask = np.zeros_like(output_predictions, dtype=np.uint8)
        for original_class, semantic_class in self.semantic_map.items():
            mask = output_predictions == original_class
            if semantic_class == 'person':
                semantic_mask[mask] = 1  # Person
            elif semantic_class == 'vehicle':
                semantic_mask[mask] = 2  # Vehicle
            elif semantic_class == 'obstacle':
                semantic_mask[mask] = 3  # Obstacle
            else:
                semantic_mask[mask] = 0  # Background
        
        return semantic_mask
    
    def colorize_segmentation(self, seg_mask):
        """Create colored visualization of segmentation"""
        h, w = seg_mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Color map
        colors = {
            0: [50, 50, 50],      # Background - dark gray
            1: [255, 100, 100],   # Person - red
            2: [100, 100, 255],   # Vehicle - blue
            3: [255, 200, 100],   # Obstacle - orange
        }
        
        for class_id, color in colors.items():
            colored[seg_mask == class_id] = color
        
        return colored

def main(args=None):
    rclpy.init(args=args)
    node = SemanticSegmentationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()