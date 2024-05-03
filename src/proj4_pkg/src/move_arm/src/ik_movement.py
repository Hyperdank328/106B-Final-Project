#!/usr/bin/env python
import rospy
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest, GetPositionIKResponse
from geometry_msgs.msg import PoseStamped
from moveit_commander import MoveGroupCommander
from intera_interface import gripper as robot_gripper
import numpy as np
from numpy import linalg
import sys


def move_to_position(position, rotation, service):
    input('Press [ Enter ]: ')
    # Construct the request
    request = GetPositionIKRequest()
    request.ik_request.group_name = "right_arm"

    # If a Sawyer does not have a gripper, replace '_gripper_tip' with '_wrist' instead
    link = "_gripper_tip" # TODO: check the frames on the robot in RVIZ

    request.ik_request.ik_link_name = link
    # request.ik_request.attempts = 20
    request.ik_request.pose_stamped.header.frame_id = "base"

    # Set the desired orientation for the end effector HERE
    request.ik_request.pose_stamped.pose.position.x = position[0]
    request.ik_request.pose_stamped.pose.position.y = position[1]
    request.ik_request.pose_stamped.pose.position.z = position[2]
    request.ik_request.pose_stamped.pose.orientation.x = rotation[0]
    request.ik_request.pose_stamped.pose.orientation.y = rotation[1]
    request.ik_request.pose_stamped.pose.orientation.z = rotation[2]
    request.ik_request.pose_stamped.pose.orientation.w = rotation[3]
    try:
        # Send the request to the service
        response = service(request)

        # Print the response HERE
        print(response)
        group = MoveGroupCommander("right_arm")

        # Setting position and orientation target
        group.set_pose_target(request.ik_request.pose_stamped)

        # TRY THIS
        # Setting just the position without specifying the orientation
        # group.set_position_target([0.5, 0.5, 0.0])

        # Plan IK
        plan = group.plan()
        user_input = input(
            "Enter 'y' if the trajectory looks safe on RVIZ")

        # Execute IK if safe
        if user_input == 'y':
            group.execute(plan[1])

    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


def main():
    # Wait for the IK service to become available
    rospy.wait_for_service('compute_ik')
    rospy.init_node('service_query')
    # Create the function used to call the service
    compute_ik = rospy.ServiceProxy('compute_ik', GetPositionIK)

    x = [0.861, 0.768, 0.746, 0.662]
    y = [-0.303, -0.131, 0.736, 0.681]
    z = [-0.0805, 0.029, 0.124, -0.110]

    # rot_x = [0.680, 0.679, -0.001, 0.694]
    # rot_y = [0.733, 0.734, 0.683, 0.720]
    # rot_z = [-0.011, 0.008, 0.014, 0.017]
    # rot_w = [-0.003, -0.014, 0.730, -0.014]
    rot_x = [0.0, 0.0, 0.0, 0.0]
    rot_y = [1.0, 1.0, 1.0, 1.0]
    rot_z = [0.0, 0.0, 0.0, 0.0]
    rot_w = [0.0, 0.0, 0.0, 0.0]

    rot = [0.0, 1.0, 0.0, 0.0]

    p1 = [0.861, -0.303, -0.085]
    p2 = [0.854, -0.293, -0.165]
    p3 = [0.962, 0.079, -0.088]
    p4 = [0.959, 0.086, -0.160]
    right_gripper = robot_gripper.Gripper('right_gripper')
    right_gripper.calibrate()

    i = 0
    while not rospy.is_shutdown():
        move_to_position(p1, rot, compute_ik)
        move_to_position(p2, rot, compute_ik)
        right_gripper.close()
        rospy.sleep(1.0)
        move_to_position(p1, rot, compute_ik)
        move_to_position(p3, rot, compute_ik)
        move_to_position(p4, rot, compute_ik)
        right_gripper.open()
        rospy.sleep(1.0)


# Python's syntax for a main() method
if __name__ == '__main__':
    main()
