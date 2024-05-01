#!/usr/bin/env python
import rospy
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest, GetPositionIKResponse
from geometry_msgs.msg import PoseStamped
from moveit_commander import MoveGroupCommander
import numpy as np
from numpy import linalg
import sys
from intera_interface import gripper as robot_gripper
import tf2_ros as tf
import tf.transformations as tr

from movement import *
from grasping import *

X_OFFSET = -0.005
Y_OFFSET = -0.210
Z_OFFSET = 0.115

# grav pawn 3 y+0.015 z-0.02
# grav pawn 4 z-0.025

metric = "rfc"
num = 4
obj = 'nozzle'

def main():
    # Wait for the IK service to become available
    rospy.wait_for_service('compute_ik')
    rospy.init_node('service_query')


    # Set up the right gripper
    right_gripper = robot_gripper.Gripper('right_gripper')

    # Calibrate the gripper (other commands won't work unless you do this first)
    # print('Calibrating...')
    # right_gripper.calibrate()
    # rospy.sleep(0.5)

    # Create the function used to call the service
    compute_ik = rospy.ServiceProxy('compute_ik', GetPositionIK)
    
    # +--------------------------------------------------------+
    # | Get the orientation of the gripper where we grip
    # +--------------------------------------------------------+
    
    # Denotes the position to grasp the object
    #TODO: Complete function
    target = get_grasp_location(9)
    # target = calibrate_grasp_location(9)
    
    # Denotes the position when the object is lifted
    above = target.copy()
    above[2] += 0.1
    lifted = target.copy()
    lifted[2] += 0.2
    

    # +--------------------------------------------------------+
    # | Run the loop...
    # +--------------------------------------------------------+
    # input('Press [ Enter ]: ')
        
        
    # Open gripper
    print('Opening...')
    right_gripper.open()
    rospy.sleep(1.0)

    # Move above object
    print('Moving above object... ')
    movepos(above, compute_ik)

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

    # Go back up
    print('Leaving object... ')
    movepos(lifted, compute_ik)


    print('Done!')
        
        
#TODO: Complete function
def get_grasp_location(tag_number):
    
    # Get location of ar tag and middle of the target
    tfbuffer = tf.Buffer()
    tfListener = tf.TransformListener(tfbuffer)
    try:
        trans = tfbuffer.lookup_transform("base", "ar_marker_"+str(tag_number), rospy.Time(), rospy.Duration(5))
        # print("Transform:")
        # print(trans)
    except Exception as e:
        print(e)
        print("Failed to get transform ...")
        exit(0)
        
    
    tag_pos = [getattr(trans.transform.translation, dim) for dim in ('x', 'y', 'z')]
    tag_pos[0] += X_OFFSET
    tag_pos[1] += Y_OFFSET
    tag_pos[2] += Z_OFFSET
    # print("Tag Pos:", tag_pos)
    vertices, normals, poses, results = load_grasp_data(obj)
    mesh = load_mesh(obj)
    pose = custom_grasp_planner(mesh, vertices[num], metric, num)
    transformation_matrix = pose["pose"]
    visualize_grasp(mesh, vertices[num], transformation_matrix)
    point = transformation_matrix[3,0:3]
    quat = tr.quaternion_from_matrix(transformation_matrix)
    
    # quat = [0,1,0,0]
    #TODO: Complete the grasp stuff
    grasp_loc = [point[0] + tag_pos[0], point[1] + tag_pos[1], point[2] + tag_pos[2], quat[0], quat[1], quat[2], quat[3]]
    # print("Grasp Location:")
    # print(grasp_loc[0:2])
    # grasp_loc = [tag_pos[0], tag_pos[1], tag_pos[2], 0, 0, 0, 0]
    
    # print("Grasp Location:")
    # print(grasp_loc)
    print("Metric Value: " + str(pose["metric"]) + " from the " + metric + " metric")
    return grasp_loc
    
def calibrate_grasp_location(tag_number):
    
    
    # Get location of ar tag and middle of the transformation_matrix target
    tfbuffer = tf.Buffer()
    tfListener = tf.TransformListener(tfbuffer)
    try:
        trans = tfbuffer.lookup_transform("base", "ar_marker_"+str(tag_number), rospy.Time(), rospy.Duration(5))
        # print("Transform:")
        # print(trans)
    except Exception as e:
        print(e)
        print("Failed to get transform ...")
        exit(0)
        
    
    tag_pos = [getattr(trans.transform.translation, dim) for dim in ('x', 'y', 'z')]
    tag_pos[0] += X_OFFSET
    tag_pos[1] += Y_OFFSET
    tag_pos[2] += Z_OFFSET
    
    grasp_loc = [tag_pos[0], tag_pos[1], tag_pos[2], 0, 1, 0, 0]
    
    print("Grasp Location:")
    print(grasp_loc)
    return grasp_loc

if __name__ == '__main__':
    main()