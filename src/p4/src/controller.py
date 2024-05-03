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
from std_msgs.msg import Float32

from movement import *
from grasping import *

X_OFFSET = 0.00 # -0.005
Y_OFFSET = 0.05 # -0.210
Z_OFFSET = 0.10 # 0.115

# grav pawn 3 y+0.015 z-0.02
# grav pawn 4 z-0.025

metric = "rfc"
num = 4
obj = 'nozzle'

def main():
    # Wait for the IK service to become available
    rospy.wait_for_service('compute_ik')
    rospy.init_node('service_query')

    half = 1 # TODO: CHANGE IT
    height_sample = []
    def height_callback(msg):
        # print("DEBUG: ", type(msg.data), msg.data)
        height_sample.append(msg.data)

    height_sub = rospy.Subscriber('/height_averaged', Float32, height_callback)
    
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
    # target = get_grasp_location(15)
    # target = calibrate_grasp_location(15)
    target = np.array([ # hardcoded from AR Tag calibration
        0.7639558675535222, 
        0.17803788332915643, 
        -0.043244710353069676, 
        0.0, 
        0.6946583704589973, 
        0.0, 
        0.7193398003386512
    ])
    
    # Denotes the position when the object is lifted
    above = target.copy() # NOTE: Above is the new grasping position
    above[2] += 0.0
    lifted = above.copy()
    lifted[2] += 0.1
    measure = above.copy()
    measure[0] -= 0.1
    measure[2] += 0.05
    measure_high = measure.copy()
    measure_high[2] += 0.15
    print("AR Tag:", target)
    print("Above:", above)
    print("Lifted:", lifted)
    print("Measure:", measure)
    print("High Measure:", measure_high)
    

    # +--------------------------------------------------------+
    # | Run the loop...
    # +--------------------------------------------------------+
    # input('Press [ Enter ]: ')
        
    if half < 1:
        # Open gripper
        print('Opening...')
        right_gripper.open()
        rospy.sleep(1.0)

        print('Moving to measure position... ')
        movepos(measure_high, compute_ik, safety=False)
        rospy.sleep(0.1)
        movepos(measure, compute_ik)

        print('Measuring...')
        rospy.sleep(2.0)
        fluid_height = height_sample[-1]
        print("measured: ", fluid_height)
        measure_ok = input("proceed?")
        while not measure_ok == 'y': 
            fluid_height = height_sample[-1]
            print("measured: ", fluid_height)
            measure_ok = input("proceed?")

        print('Moving to grasp object... ')
        movepos(above, compute_ik)

        print('Closing...')
        right_gripper.close()
        rospy.sleep(1.0)

        # Lift object
        print('Lifting object... ')
        movepos(lifted, compute_ik)

        print("fluid height: ", fluid_height)

    else:
        print('End pouring... ')
        movepos(lifted, compute_ik)

        print('Lowering object... ')
        movepos(above, compute_ik)
        
        print('Opening...')
        right_gripper.open()
        rospy.sleep(1.0)

        print('Returning to measure... ')
        movepos(measure, compute_ik)
        measure_high[2] += 0.1
        movepos(measure_high, compute_ik)

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

    deg = (88 / 180) * np.pi
    transformation_matrix = np.array([
        [np.cos(deg), 0, np.sin(deg), 0],
        [0, 1, 0, 0],
        [-np.sin(deg), 0, np.cos(deg), 0], 
        [0, 0, 0, 1]
    ])
    quat = tr.quaternion_from_matrix(transformation_matrix)
    
    grasp_loc = [tag_pos[0], tag_pos[1], tag_pos[2], quat[0], quat[1], quat[2], quat[3]]
    
    print("Grasp Location:")
    print(grasp_loc)
    return grasp_loc

if __name__ == '__main__':
    main()

# def euler_to_quaternion(yaw, pitch, roll):
    #     qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    #     qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    #     qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    #     qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    #     return [qx, qy, qz, qw]

    # def helper_xroll(deg):
    #     # deg = (deg / 180) * np.pi
    #     transformation_matrix = np.array([
    #         [1, 0, 0, 0],
    #         [0, np.cos(deg), -np.sin(deg), 0],
    #         [0, np.sin(deg), np.cos(deg), 0], 
    #         [0, 0, 0, 1]
    #     ])
    #     # transformation_matrix = np.array([
    #     #     [np.cos(deg), -np.sin(deg), 0, 0],
    #     #     [np.sin(deg), np.cos(deg), 0, 0], 
    #     #     [0, 0, 1, 0],
    #     #     [0, 0, 0, 1]
    #     # ])
    #     return tr.quaternion_from_matrix(transformation_matrix)
        

    # L = 10
    # D = 6.5
    # # h = 6.5 * fluid_height / 380
    # h = 3
    # theta_0 = np.pi / 2 # -np.arctan(L ** 2 / 2 * D * h)
    # theta_1 = np.pi # -np.arctan(L ** 2 / D * h)
    # print("ANGLES: ", theta_0, theta_1)

    # # start = lifted.copy()
    # quat = helper_xroll(theta_0)
    # start = [lifted[0], lifted[1], lifted[2], quat[0], quat[1], quat[2], quat[3]]
    # print('Going to initial pouring angle... ')
    # movepos(start, compute_ik, safety=False)

    # # end = start.copy()
    # quat = helper_xroll(theta_1)
    # end = [lifted[0], lifted[1], lifted[2], quat[0], quat[1], quat[2], quat[3]]
    # print('Pouring... ')
    # movepos(end, compute_ik, safety=False)