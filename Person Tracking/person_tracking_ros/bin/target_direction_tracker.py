#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32

# Define thresholds
LEFT_THRESHOLD = -15
RIGHT_THRESHOLD = 15
AWAY_THRESHOLD = 25

# Variable to store latest target angle
target_angle = 0.0

def angle_callback(msg):
    global target_angle
    target_angle = msg.data  # Update angle
    direction = determine_direction(target_angle)
    rospy.loginfo(f"Target Angle: {target_angle}° → Direction: {direction}")

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
    rospy.init_node("target_angle_tracker")
    rospy.Subscriber("/person_tracking/target_angle", Float32, angle_callback)
    rospy.spin()
