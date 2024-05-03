#! /usr/bin/env python

import rospy
import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from intera_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest


def ik_service_client():
    service_name = "ExternalTools/right/PositionKinematicsNode/IKService"
    ik_service_proxy = rospy.ServiceProxy(service_name, SolvePositionIK)
    ik_request = SolvePositionIKRequest()
    header = Header(stamp=rospy.Time.now(), frame_id='base')

    # Create a PoseStamped and specify header (specifying a header is very important!)
    pose_stamped = PoseStamped()
    pose_stamped.header = header

    # Set end effector position: YOUR CODE HERE
    # Prompt user for x, y, and z position input
    pose_stamped.pose.position.x = float(input("Enter x position: "))
    pose_stamped.pose.position.y = float(input("Enter y position: "))
    pose_stamped.pose.position.z = float(input("Enter z position: "))

    # Set end effector quaternion: YOUR CODE HERE
    # Prompt the user for roll, pitch, and yaw angles
    roll = float(input("Enter roll angle: "))
    pitch = float(input("Enter pitch angle: "))
    yaw = float(input("Enter yaw angle: "))
    # Convert roll, pitch, and yaw to quaternions
    r = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    r = r.as_quat()
    # Set the quaternion values
    pose_stamped.pose.orientation.x = r[0]
    pose_stamped.pose.orientation.y = r[1]
    pose_stamped.pose.orientation.z = r[2]
    pose_stamped.pose.orientation.w = r[3]
    # Make sure quaternion is normalized
    assert np.linalg.norm(r) == 1

    # Add desired pose for inverse kinematics
    ik_request.pose_stamp.append(pose_stamped)
    # Request inverse kinematics from base to "right_hand" link
    ik_request.tip_names.append('right_hand')

    rospy.loginfo("Running Simple IK Service Client example.")

    try:
        rospy.wait_for_service(service_name, 5.0)
        response = ik_service_proxy(ik_request)
    except (rospy.ServiceException, rospy.ROSException) as e:
        rospy.logerr("Service call failed: %s" % (e,))
        return

    # Check if result valid, and type of seed ultimately used to get solution
    if (response.result_type[0] > 0):
        rospy.loginfo("SUCCESS!")
        # Format solution into Limb API-compatible dictionary
        limb_joints = dict(
            list(zip(response.joints[0].name, response.joints[0].position)))
        print(limb_joints)
        rospy.loginfo("\nIK Joint Solution:\n%s", limb_joints)
        rospy.loginfo("------------------")
        rospy.loginfo("Response Message:\n%s", response)
    else:
        rospy.logerr("INVALID POSE - No Valid Joint Solution Found.")
        rospy.logerr("Result Error %d", response.result_type[0])
        return False

    return True


def main():
    rospy.init_node("ik_service_client")

    ik_service_client()


if __name__ == '__main__':
    main()
