#!/usr/bin/env python
import rospy
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest, GetPositionIKResponse
from geometry_msgs.msg import PoseStamped
from moveit_commander import MoveGroupCommander
import numpy as np
from numpy import linalg
import sys
from intera_interface import gripper as robot_gripper
import tf2_ros

from movement import *
from grasping import *


def main():
    # Wait for the IK service to become available
    rospy.wait_for_service('compute_ik')
    rospy.init_node('service_query')


    # Set up the right gripper
    right_gripper = robot_gripper.Gripper('right_gripper')

    # Calibrate the gripper (other commands won't work unless you do this first)
    print('Calibrating...')
    right_gripper.calibrate()
    rospy.sleep(2.0)

    # Create the function used to call the service
    compute_ik = rospy.ServiceProxy('compute_ik', GetPositionIK)

    # +--------------------------------------------------------+
    # | Get the orientation of the gripper where we grip
    # +--------------------------------------------------------+

    # Denotes the position to grasp the object
    #TODO: Complete function
    # target = get_grasp_location(12)
    target = calibrate_grasp_location(12)

    # Denotes the position when the object is lifted
    lifted = target.copy
    lifted[2] += 0.2


    # +--------------------------------------------------------+
    # | Run the loop...
    # +--------------------------------------------------------+
    input('Press [ Enter ]: ')

    # TODO: update trajectory and movement stages
    # Open gripper
    print('Opening...')
    right_gripper.open()
    rospy.sleep(1.0)

    # Move to grasp location
    print('Moving to target position... ')
    movepos(target, compute_ik)

    # Grasp object
    print('Closing...')
    right_gripper.close()
    rospy.sleep(1.0)

    # Lift object
    print('Lifting object... ')
    movepos(lifted, compute_ik)

    rospy.sleep(1.0)

    # Move back to position
    print('Placing Object... ')
    movepos(target, compute_ik)

    # Open gripper
    print('Opening...')
    right_gripper.open()
    rospy.sleep(1.0)


    print('Done!')


#TODO: Complete function
def get_grasp_location(tag_number):

    X_OFFSET = 0.2
    Y_OFFSET = 0
    Z_OFFSET = 0.1

    # Get location of ar tag and middle of the target
    tfbuffer = tf2_ros.Buffer()
    tfListener = tf2_ros.TransformListener(tfbuffer)
    try:
        trans = tfbuffer.lookup_transform("base", "ar_marker_"+str(tag_number), rospy.Time(), rospy.Duration(5))
        print("Transform:")
        print(trans)
    except Exception as e:
        print(e)
        print("Failed to get transform ...")
        exit(0)


    tag_pos = [getattr(trans.transform.translation, dim) for dim in ('x', 'y', 'z')]
    tag_pos[0] += X_OFFSET
    tag_pos[1] += Y_OFFSET
    tag_pos[2] += Z_OFFSET

    #TODO: Complete the grasp stuff (can fix in grasping or just change it here, the grasp should be constant back2front horizontal grip)
    grasp_loc = get_gripper_pose(obj="pawn", metric="grav")
    # grasp_loc = [tag_pos[0], tag_pos[1], tag_pos[2], 0, 0, 0, 0]

    print("GRasp Location:")
    print(grasp_loc)
    return grasp_loc

def calibrate_grasp_location(tag_number):

    X_OFFSET = 0.2
    Y_OFFSET = 0
    Z_OFFSET = 0.1

    # Get location of ar tag and middle of the target
    tfbuffer = tf2_ros.Buffer()
    tfListener = tf2_ros.TransformListener(tfbuffer)
    try:
        trans = tfbuffer.lookup_transform("base", "ar_marker_"+str(tag_number), rospy.Time(), rospy.Duration(5))
        print("Transform:")
        print(trans)
    except Exception as e:
        print(e)
        print("Failed to get transform ...")
        exit(0)


    tag_pos = [getattr(trans.transform.translation, dim) for dim in ('x', 'y', 'z')]
    tag_pos[0] += X_OFFSET
    tag_pos[1] += Y_OFFSET
    tag_pos[2] += Z_OFFSET

    #TODO: Complete the grasp stuff
    obj = "pawn"
    metric = "grav"
    grasp_loc = get_gripper(obj=obj, metric=metric)
    # grasp_loc = [tag_pos[0], tag_pos[1], tag_pos[2], 0, 0, 0, 0]

    print("GRasp Location:")
    print(grasp_loc)
    return grasp_loc


if __name__ == '__main__':
    main()
