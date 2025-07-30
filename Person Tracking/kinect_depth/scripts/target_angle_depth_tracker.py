#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import cv2
import os

# Define angle thresholds
LEFT_THRESHOLD = -10
RIGHT_THRESHOLD = 10

# Variable to store latest target angle
target_angle = 0.0
last_direction = None  # To detect direction change

# Initialize CvBridge
bridge = CvBridge()

# Path to output file
output_file_path = "/home/paula/catkin_ws/src/kinect_depth/scripts/direction_output.txt"  # CHANGE to your path

def angle_callback(msg):
    global target_angle
    target_angle = msg.data

def depth_callback(depth_msg):
    global last_direction
    try:
        depth_image = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        center_pixel = depth_image[depth_image.shape[0] // 2, depth_image.shape[1] // 2]
        depth_in_meters = center_pixel / 1000.0

        direction = determine_direction(target_angle, depth_in_meters)

        if direction != last_direction:
            last_direction = direction
            rospy.loginfo(f"Target Angle: {target_angle}°, Depth: {depth_in_meters:.2f} m → Direction: {direction}")

            # Save new direction to file
            with open(output_file_path, "w") as file:
                file.write(direction)

    except Exception as e:
        rospy.logerr(f"Error processing depth image: {e}")

def determine_direction(angle, depth):
    if ((depth > 2.0) and (angle < LEFT_THRESHOLD)):
        return "AWAY LEFT"
    elif ((depth > 2.0) and (angle > RIGHT_THRESHOLD)):
        return "AWAY RIGHT"
    elif ((depth < 2.0) and (angle < LEFT_THRESHOLD)):
        return "CENTER LEFT"
    elif ((depth < 2.0) and (angle > RIGHT_THRESHOLD)):
        return "CENTER RIGHT"
    elif depth < 2.0:
        return "CENTER"
    else:
        return "AWAY"

if __name__ == "__main__":
    rospy.init_node("target_angle_depth_tracker")

    rospy.Subscriber("/person_tracking/target_angle", Float32, angle_callback)
    rospy.Subscriber("/camera/depth/image_raw", Image, depth_callback)

    rospy.spin()

