#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

# Define depth thresholds (in meters)
NEAR_THRESHOLD = 0.5   # Meters
AWAY_THRESHOLD = 3.0   # Meters

# Define pixel thresholds for left/right detection
IMAGE_WIDTH = 640  # Kinect V1 resolution width (adjust if needed)
LEFT_THRESHOLD = IMAGE_WIDTH // 3
RIGHT_THRESHOLD = 2 * IMAGE_WIDTH // 3

# Create CvBridge object
bridge = CvBridge()

def depth_callback(msg):
    try:
        # Convert the depth image message to an OpenCV image
        depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        
        # Check the shape of the depth image
        rospy.loginfo(f"Depth image shape: {depth_image.shape}")

        # Get the depth at the center of the image (assuming center pixel)
        height, width = depth_image.shape
        center_pixel_depth = depth_image[height // 2, width // 2] / 1000.0  # Convert from mm to meters
        
        # Get the x-coordinate of the center pixel
        center_x = width // 2
        
        # Log the depth at the center
        rospy.loginfo(f"Depth at center pixel: {center_pixel_depth} meters")
        
        # Determine the direction based on depth and position
        depth_direction = determine_depth_direction(center_pixel_depth)
        horizontal_direction = determine_horizontal_direction(center_x)
        
        # Combine the results
        direction = f"{depth_direction} - {horizontal_direction}"
        
        # Log final direction
        rospy.loginfo(f"Target Distance: {center_pixel_depth} meters → Direction: {direction}")
        
    except Exception as e:
        rospy.logerr(f"Error in processing depth image: {e}")

def determine_depth_direction(depth):
    """Determine depth-based direction"""
    if depth > AWAY_THRESHOLD:
        return "AWAY"
    elif depth < NEAR_THRESHOLD:
        return "NEAR"
    else:
        return "CENTER"

def determine_horizontal_direction(center_x):
    """Determine horizontal direction based on pixel position"""
    if center_x < LEFT_THRESHOLD:
        return "LEFT"
    elif center_x > RIGHT_THRESHOLD:
        return "RIGHT"
    else:
        return "CENTER"

if __name__ == "__main__":
    # Initialize the ROS node
    rospy.init_node("target_depth_tracker")
    
    # Subscribe to the depth image topic
    rospy.Subscriber("/camera/depth/image_raw", Image, depth_callback)
    
    # Keep the program running
    rospy.spin()
