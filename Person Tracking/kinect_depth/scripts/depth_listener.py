#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import cv2

# Define thresholds (adjusted)
LEFT_THRESHOLD = -10   # Left angle threshold in degrees
RIGHT_THRESHOLD = 10   # Right angle threshold in degrees
AWAY_THRESHOLD = 15    # Angle threshold for "AWAY" state (in degrees)

# Variable to store latest target angle
target_angle = 0.0

# Initialize CvBridge
bridge = CvBridge()

def angle_callback(msg):
    global target_angle
    target_angle = msg.data  # Update angle
    direction = determine_direction(target_angle)
    rospy.loginfo(f"Target Angle: {target_angle}° → Direction: {direction}")

def depth_callback(depth_msg):
    """Callback to process depth data"""
    try:
        # Convert depth image to OpenCV format
        depth_image = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

        # Get the depth at the center pixel (i.e., middle of the image)
        center_pixel = depth_image[depth_image.shape[0] // 2, depth_image.shape[1] // 2]

        # Calculate the distance of the target using the center pixel depth
        rospy.loginfo(f"Depth at center pixel: {center_pixel} meters")
        
        # Use the angle information to log both depth and direction
        direction = determine_direction(target_angle)
        rospy.loginfo(f"Direction: {direction}, Depth at center: {center_pixel} meters")

    except Exception as e:
        rospy.logerr(f"Error processing depth image: {e}")

def determine_direction(angle):
    """Determine direction based on angle thresholds"""
    if abs(angle) > AWAY_THRESHOLD:
        return "AWAY"
    elif angle < LEFT_THRESHOLD:
        return "LEFT"
    elif angle > RIGHT_THRESHOLD:
        return "RIGHT"
    else:
        return "CENTER"

if __name__ == "__main__":
    rospy.init_node("target_angle_depth_tracker")

    # Subscribe to the target angle and depth image topics
    rospy.Subscriber("/person_tracking/target_angle", Float32, angle_callback)
    rospy.Subscriber("/camera/depth/image_raw", Image, depth_callback)

    rospy.spin()

