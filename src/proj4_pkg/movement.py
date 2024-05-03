#!/usr/bin/env python
import rospy
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest, GetPositionIKResponse
from geometry_msgs.msg import PoseStamped
from moveit_commander import MoveGroupCommander
import numpy as np
from numpy import linalg
import sys
from intera_interface import gripper as robot_gripper
from tf.transformations import quaternion_from_euler

OFFSET_X = 0
OFFSET_Y = 0
OFFSET_Z = 0


def get_position(tfBuffer, tag):
        try:
            trans = tfBuffer.lookup_transform('base', 'ar_marker_{0}'.format(tag), rospy.Time(0), rospy.Duration(10.0))
        except Exception as e:
            print(e)
        target_pos = np.array([getattr(trans.transform.translation, dim) for dim in ('x', 'y', 'z')])
        target_pos[0] += OFFSET_X
        target_pos[1] += OFFSET_Y
        target_pos[2] += OFFSET_Z
        return [target_pos[0] + OFFSET_X, target_pos[1] + OFFSET_Y, target_pos[2] + OFFSET_Z]


def movepos(target, compute_ik):
    # Construct the request
    request = GetPositionIKRequest()
    request.ik_request.group_name = "right_arm"

    # If a Sawyer does not have a gripper, replace '_gripper_tip' with '_wrist' instead
    link = "right_gripper_tip"

    request.ik_request.ik_link_name = link
    # request.ik_request.attempts = 20
    request.ik_request.pose_stamped.header.frame_id = "base"
    

    # Set the desired orientation for the end effector HERE
    request.ik_request.pose_stamped.pose.position.x = target[0]
    request.ik_request.pose_stamped.pose.position.y = target[1]
    request.ik_request.pose_stamped.pose.position.z = target[2] 
    request.ik_request.pose_stamped.pose.orientation.x = target[3]
    request.ik_request.pose_stamped.pose.orientation.y = target[4]
    request.ik_request.pose_stamped.pose.orientation.z = target[5]
    request.ik_request.pose_stamped.pose.orientation.w = target[6]
        
    try:
        # Send the request to the service
        response = compute_ik(request)
        # print(request)
            
        # Print the response HERE
        # print(response)
        group = MoveGroupCommander("right_arm")

        # Setting position and orientation target
        group.set_pose_target(request.ik_request.pose_stamped)

        # Plan IK
        plan = group.plan()
        user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")
            
        # Execute IK if safe
        if user_input == 'y':
            group.execute(plan[1])
            
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)