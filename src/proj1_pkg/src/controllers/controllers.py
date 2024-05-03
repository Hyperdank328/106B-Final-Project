#!/usr/bin/env python

"""
Starter script for Project 1B. 
Author: Chris Correa, Valmik Prabhu
"""

# Python imports
import sys
import numpy as np
import itertools
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Lab imports
from utils.utils import *
from controllers.controller_utils import *

from paths.paths import MotionPath
from paths.trajectories import LinearTrajectory
from trac_ik_python.trac_ik import IK
from sawyer_pykdl import sawyer_kinematics
from path_planner import PathPlanner
import moveit_commander

# ROS imports
try:
    import tf
    import tf2_ros
    import rospy
    import baxter_interface
    import intera_interface
    from geometry_msgs.msg import PoseStamped
    from moveit_msgs.msg import RobotTrajectory
except:
    pass

NUM_JOINTS = 7

class Controller:

    def __init__(self, limb, kin):
        """
        Constructor for the superclass. All subclasses should call the superconstructor

        Parameters
        ----------
        limb : :obj:`baxter_interface.Limb` or :obj:`intera_interface.Limb`
        kin : :obj:`baxter_pykdl.baxter_kinematics` or :obj:`sawyer_pykdl.sawyer_kinematics`
            must be the same arm as limb
        """

        # Run the shutdown function when the ros node is shutdown
        rospy.on_shutdown(self.shutdown)
        self._limb = limb
        self._kin = kin

        # Set this attribute to True if the present controller is a jointspace controller.
        self.is_jointspace_controller = False

    def step_control(self, target_position, target_velocity, target_acceleration):
        """
        makes a call to the robot to move according to it's current position and the desired position 
        according to the input path and the current time. Each Controller below extends this 
        class, and implements this accordingly.  

        Parameters
        ----------
        target_position : 7x' or 6x' :obj:`numpy.ndarray` 
            desired positions
        target_velocity : 7x' or 6x' :obj:`numpy.ndarray` 
            desired velocities
        target_acceleration : 7x' or 6x' :obj:`numpy.ndarray` 
            desired accelerations
        """
        pass

    def interpolate_path(self, path, t, current_index = 0):
        """
        interpolates over a :obj:`moveit_msgs.msg.RobotTrajectory` to produce desired
        positions, velocities, and accelerations at a specified time

        Parameters
        ----------
        path : :obj:`moveit_msgs.msg.RobotTrajectory`
        t : float
            the time from start
        current_index : int
            waypoint index from which to start search

        Returns
        -------
        target_position : 7x' or 6x' :obj:`numpy.ndarray` 
            desired positions
        target_velocity : 7x' or 6x' :obj:`numpy.ndarray` 
            desired velocities
        target_acceleration : 7x' or 6x' :obj:`numpy.ndarray` 
            desired accelerations
        current_index : int
            waypoint index at which search was terminated 
        """

        # a very small number (should be much smaller than rate)
        epsilon = 0.0001

        max_index = len(path.joint_trajectory.points)-1

        # If the time at current index is greater than the current time,
        # start looking from the beginning
        if (path.joint_trajectory.points[current_index].time_from_start.to_sec() > t):
            current_index = 0

        # Iterate forwards so that you're using the latest time
        while (
            not rospy.is_shutdown() and \
            current_index < max_index and \
            path.joint_trajectory.points[current_index+1].time_from_start.to_sec() < t+epsilon
        ):
            current_index = current_index+1

        # Perform the interpolation
        if current_index < max_index:
            time_low = path.joint_trajectory.points[current_index].time_from_start.to_sec()
            time_high = path.joint_trajectory.points[current_index+1].time_from_start.to_sec()

            target_position_low = np.array(
                path.joint_trajectory.points[current_index].positions
            )
            target_velocity_low = np.array(
                path.joint_trajectory.points[current_index].velocities
            )
            target_acceleration_low = np.array(
                path.joint_trajectory.points[current_index].accelerations
            )

            target_position_high = np.array(
                path.joint_trajectory.points[current_index+1].positions
            )
            target_velocity_high = np.array(
                path.joint_trajectory.points[current_index+1].velocities
            )
            target_acceleration_high = np.array(
                path.joint_trajectory.points[current_index+1].accelerations
            )

            target_position = target_position_low + \
                (t - time_low)/(time_high - time_low)*(target_position_high - target_position_low)
            target_velocity = target_velocity_low + \
                (t - time_low)/(time_high - time_low)*(target_velocity_high - target_velocity_low)
            target_acceleration = target_acceleration_low + \
                (t - time_low)/(time_high - time_low)*(target_acceleration_high - target_acceleration_low)

        # If you're at the last waypoint, no interpolation is needed
        else:
            target_position = np.array(path.joint_trajectory.points[current_index].positions)
            target_velocity = np.array(path.joint_trajectory.points[current_index].velocities)
            target_acceleration = np.array(path.joint_trajectory.points[current_index].velocities)

        return (target_position, target_velocity, target_acceleration, current_index)


    def shutdown(self):
        """
        Code to run on shutdown. This is good practice for safety
        """
        rospy.loginfo("Stopping Controller")

        # Set velocities to zero
        self.stop_moving()
        rospy.sleep(0.1)

    def stop_moving(self):
        """
        Set robot joint velocities to zero
        """
        zero_vel_dict = joint_array_to_dict(np.zeros(NUM_JOINTS), self._limb)
        self._limb.set_joint_velocities(zero_vel_dict)

    def plot_results(
        self,
        times,
        actual_positions, 
        actual_velocities, 
        target_positions, 
        target_velocities
    ):
        """
        Plots results.
        If the path is in joint space, it will plot both workspace and jointspace plots.
        Otherwise it'll plot only workspace

        Inputs:
        times : nx' :obj:`numpy.ndarray`
        actual_positions : nx7 or nx6 :obj:`numpy.ndarray`
            actual joint positions for each time in times
        actual_velocities: nx7 or nx6 :obj:`numpy.ndarray`
            actual joint velocities for each time in times
        target_positions: nx7 or nx6 :obj:`numpy.ndarray`
            target joint or workspace positions for each time in times
        target_velocities: nx7 or nx6 :obj:`numpy.ndarray`
            target joint or workspace velocities for each time in times
        """

        # Make everything an ndarray
        times = np.array(times)
        actual_positions = np.array(actual_positions)
        actual_velocities = np.array(actual_velocities)
        target_positions = np.array(target_positions)
        target_velocities = np.array(target_velocities)

        # Find the actual workspace positions and velocities
        actual_workspace_positions = np.zeros((len(times), 3))
        actual_workspace_velocities = np.zeros((len(times), 3))
        actual_workspace_quaternions = np.zeros((len(times), 4))

        for i in range(len(times)):
            positions_dict = joint_array_to_dict(actual_positions[i], self._limb)
            fk = self._kin.forward_position_kinematics(joint_values=positions_dict)
            
            actual_workspace_positions[i, :] = fk[:3]
            actual_workspace_velocities[i, :] = \
                self._kin.jacobian(joint_values=positions_dict)[:3].dot(actual_velocities[i])
            actual_workspace_quaternions[i, :] = fk[3:]
        # check if joint space
        if self.is_jointspace_controller:
            # it's joint space

            target_workspace_positions = np.zeros((len(times), 3))
            target_workspace_velocities = np.zeros((len(times), 3))
            target_workspace_quaternions = np.zeros((len(times), 4))

            for i in range(len(times)):
                positions_dict = joint_array_to_dict(target_positions[i], self._limb)
                target_workspace_positions[i, :] = \
                    self._kin.forward_position_kinematics(joint_values=positions_dict)[:3]
                target_workspace_velocities[i, :] = \
                    self._kin.jacobian(joint_values=positions_dict)[:3].dot(target_velocities[i])
                target_workspace_quaternions[i, :] = np.array([0, 1, 0, 0])

            # Plot joint space
            plt.figure()
            joint_num = len(self._limb.joint_names())
            for joint in range(joint_num):
                plt.subplot(joint_num,2,2*joint+1)
                plt.plot(times, actual_positions[:,joint], label='Actual')
                plt.plot(times, target_positions[:,joint], label='Desired')
                plt.xlabel("Time (t)")
                plt.ylabel("Joint " + str(joint) + " Position Error")
                plt.legend()

                plt.subplot(joint_num,2,2*joint+2)
                plt.plot(times, actual_velocities[:,joint], label='Actual')
                plt.plot(times, target_velocities[:,joint], label='Desired')
                plt.xlabel("Time (t)")
                plt.ylabel("Joint " + str(joint) + " Velocity Error")
                plt.legend()
            print("Close the plot window to continue")
            plt.show()

        else:
            # it's workspace
            target_workspace_positions = target_positions
            target_workspace_velocities = target_velocities
            target_workspace_quaternions = np.zeros((len(times), 4))
            target_workspace_quaternions[:, 1] = 1

        plt.figure()
        workspace_joints = ('X', 'Y', 'Z')
        joint_num = len(workspace_joints)
        for joint in range(joint_num):
            plt.subplot(joint_num,2,2*joint+1)
            plt.plot(times, actual_workspace_positions[:,joint], label='Actual')
            plt.plot(times, target_workspace_positions[:,joint], label='Desired')
            plt.xlabel("Time (t)")
            plt.ylabel(workspace_joints[joint] + " Position Error")
            plt.legend()

            plt.subplot(joint_num,2,2*joint+2)
            plt.plot(times, actual_velocities[:,joint], label='Actual')
            plt.plot(times, target_velocities[:,joint], label='Desired')
            plt.xlabel("Time (t)")
            plt.ylabel(workspace_joints[joint] + " Velocity Error")
            plt.legend()
        # would mean that this "angle error" is always zero.
        angles = []
        for t in range(len(times)):
            quat1 = target_workspace_quaternions[t]
            quat2 = actual_workspace_quaternions[t]
            theta = axis_angle(quat1, quat2)
            angles.append(theta)

        plt.figure()
        plt.plot(times, angles)
        plt.xlabel("Time (s)")
        plt.ylabel("Error Angle of End Effector (rad)")
        print("Close the plot window to continue")
        plt.show()
        

    def execute_path(self, path, rate=200, timeout=None, log=False):
        """
        takes in a path and moves the baxter in order to follow the path.  

        Parameters
        ----------
        path : :obj:`moveit_msgs.msg.RobotTrajectory`
        rate : int
            This specifies how many ms between loops.  It is important to
            use a rate and not a regular while loop because you want the
            loop to refresh at a constant rate, otherwise you would have to
            tune your PD parameters if the loop runs slower / faster
        timeout : int
            If you want the controller to terminate after a certain number
            of seconds, specify a timeout in seconds.
        log : bool
            whether or not to display a plot of the controller performance

        Returns
        -------
        bool
            whether the controller completes the path or not
        """

        # For plotting
        if log:
            times = list()
            actual_positions = list()
            actual_velocities = list()
            target_positions = list()
            target_velocities = list()

        # For interpolation
        max_index = len(path.joint_trajectory.points)-1
        current_index = 0
        print(max_index)

        # For timing
        start_t = rospy.Time.now()
        r = rospy.Rate(rate)

        while not rospy.is_shutdown():
            # Find the time from start
            t = (rospy.Time.now() - start_t).to_sec()

            # If the controller has timed out, stop moving and return false
            if timeout is not None and t >= timeout:
                # Set velocities to zero
                self.stop_moving()
                return False

            current_position = get_joint_positions(self._limb)
            current_velocity = get_joint_velocities(self._limb)

            # Get the desired position, velocity, and effort
            (
                target_position, 
                target_velocity, 
                target_acceleration, 
                current_index
            ) = self.interpolate_path(path, t, current_index)

            # For plotting
            if log:
                times.append(t)
                actual_positions.append(current_position)
                actual_velocities.append(current_velocity)
                target_positions.append(target_position)
                target_velocities.append(target_velocity)

            # Run controller
            self.step_control(target_position, target_velocity, target_acceleration)

            # Sleep for a bit (to let robot move)
            r.sleep()

            if current_index >= max_index:
                self.stop_moving()
                break

        if log:
            self.plot_results(
                times,
                actual_positions, 
                actual_velocities, 
                target_positions, 
                target_velocities
            )
        return True
    
    def get_position(self, tfBuffer, tag):
        try:
            # trans_tip = tfBuffer.lookup_transform('base', 'right_gripper_tip', rospy.Time(0), rospy.Duration(10.0))
            trans_tip = tfBuffer.lookup_transform('base', 'stp_022312TP99620_tip_1', rospy.Time(0), rospy.Duration(10.0))
            trans_tag = tfBuffer.lookup_transform('base', 'ar_marker_{0}'.format(tag), rospy.Time(0), rospy.Duration(10.0))
        except Exception as e:
            print(e)
        current_pos = np.array([getattr(trans_tip.transform.translation, dim) for dim in ('x', 'y', 'z')])
        target_pos = np.array([getattr(trans_tag.transform.translation, dim) for dim in ('x', 'y', 'z')])
        current_pos[2] += 0.1
        target_pos[2] += 0.65
        return current_pos, target_pos
    
    def get_robot_trajectory(self, current_pos, target_pos, limb, kin, ik_solver, rate):
        trajectory = LinearTrajectory(current_pos, target_pos, 0.5)
        path = MotionPath(limb, kin, ik_solver, trajectory)
        return path.to_robot_trajectory(200, True)
        # return path.to_robot_trajectory(300, False) # for workspace controller

    def follow_ar_tag(self, tag, rate=200, timeout=None, log=False):
        """
        takes in an AR tag number and follows it with the baxter's arm.  You 
        should look at execute_path() for inspiration on how to write this. 

        Parameters
        ----------
        tag : int
            which AR tag to use
        rate : int
            This specifies how many ms between loops.  It is important to
            use a rate and not a regular while loop because you want the
            loop to refresh at a constant rate, otherwise you would have to
            tune your PD parameters if the loop runs slower / faster
        timeout : int
            If you want the controller to terminate after a certain number
            of seconds, specify a timeout in seconds.
        log : bool
            whether or not to display a plot of the controller performance

        Returns
        -------
        bool
            whether the controller completes the path or not
        """
        controller_name = "jointspace"
        
        # setup log
        if log:
            times = list()
            actual_positions = list()
            actual_velocities = list()
            target_positions = list()
            target_velocities = list()
            
        # Define iksolver and tfbuffer
        ik_solver = IK("base", "right_hand")
        limb = intera_interface.Limb('right')
        kin = sawyer_kinematics('right')
        tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tfBuffer)
        
        # Set total runtime and rate
        RUN_TIME = 20 # seconds
        r = rospy.Rate(rate)
        
        # Set initial positions and path
        current_pos, target_pos = self.get_position(tfBuffer, tag)
        last_target = target_pos
        robot_trajectory = self.get_robot_trajectory(current_pos, target_pos, limb, kin, ik_solver, rate)
        # For interpolation
        max_index = len(robot_trajectory.joint_trajectory.points)-1
        current_index = 0
        print("STARTING:")
    
        
        
        # For interpolation
        max_index = len(robot_trajectory.joint_trajectory.points)-1
        print(max_index)
        current_index = 0
        # For timing
        start_t = rospy.Time.now()
        global_start = rospy.Time.now()
        # r = rospy.Rate(rate)
    
        while not rospy.is_shutdown():
            t = (rospy.Time.now() - start_t).to_sec()
            blobal_t = (rospy.Time.now() - global_start).to_sec()
            
            # If the controller has timed out, stop moving and return false
            if timeout is not None and blobal_t >= RUN_TIME:
                # Set velocities to zero
                print("finished time")
                self.stop_moving()
                break
            
            current_pos, target_pos = self.get_position(tfBuffer, tag)
            if (np.linalg.norm(last_target-target_pos) > 0.03):
                last_target = target_pos

                # print("Current Position:", current_pos)
                # print("Target Position:", target_pos)
                print("Movement detected")
                robot_trajectory = self.get_robot_trajectory(current_pos, target_pos, limb, kin, ik_solver, rate)
                
                max_index = len(robot_trajectory.joint_trajectory.points)-1
                current_index = 0
                start_t = rospy.Time.now()
            else:
                current_position = get_joint_positions(self._limb)
                current_velocity = get_joint_velocities(self._limb)

                if current_index >= max_index:
                    print("done indexing")
                    self.stop_moving()
                    if log:
                        times.append(blobal_t)
                        actual_positions.append(current_position)
                        actual_velocities.append(current_velocity)
                        target_positions.append(target_position)
                        target_velocities.append(target_velocity)
                else:
                    # Get the desired position, velocity, and effort
                    (
                        target_position, 
                        target_velocity, 
                        target_acceleration, 
                        current_index
                    ) = self.interpolate_path(robot_trajectory, t, current_index)
                    
                    # For plotting
                    if log:
                        times.append(blobal_t)
                        actual_positions.append(current_position)
                        actual_velocities.append(current_velocity)
                        target_positions.append(target_position)
                        target_velocities.append(target_velocity)
                    # Run controller
                    self.step_control(target_position, target_velocity, target_acceleration)
            
            # Sleep for a bit (to let robot move)
            r.sleep()
            
                        
        print("Done servoing")
        if log:
            self.plot_results(
                times,
                actual_positions, 
                actual_velocities, 
                target_positions, 
                target_velocities
            )


class FeedforwardJointVelocityController(Controller):
    def step_control(self, target_position, target_velocity, target_acceleration):
        """
        ParametersFeedforward
        ----------
        target_position: 7x' ndarray of desired positions
        target_velocity: 7x' ndarray of desired velocities
        target_acceleration: 7x' ndarray of desired accelerations
        """
        self._limb.set_joint_velocities(joint_array_to_dict(target_velocity, self._limb))

class WorkspaceVelocityController(Controller):
    """
    Look at the comments on the Controller class above.  The difference between this controller and the
    PDJointVelocityController is that this controller compares the baxter's current WORKSPACE position and
    velocity desired WORKSPACE position and velocity to come up with a WORKSPACE velocity command to be sent
    to the baxter.  Then this controller should convert that WORKSPACE velocity command into a joint velocity
    command and sends that to the baxter.  Notice the shape of Kp and Kv
    """
    def __init__(self, limb, kin, Kp, Kv):
        """
        Parameters
        ----------
        limb : :obj:`baxter_interface.Limb`
        kin : :obj:`BaxterKinematics`
        Kp : 6x' :obj:`numpy.ndarray`
        Kv : 6x' :obj:`numpy.ndarray`
        """
        Controller.__init__(self, limb, kin)
        self.Kp = np.diag(Kp)
        self.Kv = np.diag(Kv)
        self.is_jointspace_controller = False

    def step_control(self, target_position, target_velocity, target_acceleration):
        """
        Makes a call to the robot to move according to its current position and the desired position 
        according to the input path and the current time.
        target_position will be a 7 vector describing the desired SE(3) configuration where the first
        3 entries are the desired position vector and the next 4 entries are the desired orientation as
        a quaternion, all written in spatial coordinates.
        target_velocity is the body-frame se(3) velocity of the desired SE(3) trajectory gd(t). This velocity
        is given as a 6D Twist (vx, vy, vz, wx, wy, wz).
        This method should call self._kin.forward_position_kinematics() to get the current workspace 
        configuration and self._limb.set_joint_velocities() to set the joint velocity to something.  
        Remember that we want to track a trajectory in SE(3), and implement the controller described in the
        project document PDF.
        Parameters
        ----------
        target_position: (7,) ndarray of desired SE(3) position (px, py, pz, qx, qy, qz, qw) (position + quaternion).
        target_velocity: (6,) ndarray of desired body-frame se(3) velocity (vx, vy, vz, wx, wy, wz).
        target_acceleration: ndarray of desired accelerations (should you need this?).
        """
        current_position = get_joint_positions(self._limb)
        current_velocity = get_joint_velocities(self._limb)

        gst = get_g_fk(self._kin, self._limb, current_position)
        # gsd = get_g_fk(self._kin, self._limb, target_position)
        gsd = get_g_matrix(target_position[0:3], target_position[3:7])
        # gst = hat(get_g_vfk(self._kin, self._limb, current_velocity))
        # gsd = hat(target_velocity)
        gtd = np.linalg.inv(gst) @ gsd
        xitd = vee(log(gtd))
        xistd = adj(gst) @ xitd
        # Vsd = target_velocity * np.array([1, 1, -1, 1, 1, 1])
        Vsd = - target_velocity
        Us = self.Kp @ xistd + Vsd
        Jinv = self._kin.jacobian_pseudo_inverse(joint_array_to_dict(current_position, self._limb))
        input = Jinv @ Us
        control_input = np.array([i for i in input[0]])
        # print(control_input)
        self._limb.set_joint_velocities(joint_array_to_dict(control_input.reshape(-1), self._limb))




class PDJointVelocityController(Controller):
    """
    Look at the comments on the Controller class above.  The difference between this controller and the 
    PDJointVelocityController is that this controller turns the desired workspace position and velocity
    into desired JOINT position and velocity.  Then it compares the difference between the baxter's 
    current JOINT position and velocity and desired JOINT position and velocity to come up with a
    joint velocity command and sends that to the baxter.  notice the shape of Kp and Kv
    """
    def __init__(self, limb, kin, Kp, Kv):
        """
        Parameters
        ----------
        limb : :obj:`baxter_interface.Limb`
        kin : :obj:`BaxterKinematics`
        Kp : 7x' :obj:`numpy.ndarray`
        Kv : 7x' :obj:`numpy.ndarray`
        """
        Controller.__init__(self, limb, kin)
        self.Kp = np.diag(Kp)
        self.Kv = np.diag(Kv)
        self.is_jointspace_controller = True

    def step_control(self, target_position, target_velocity, target_acceleration):
        """
        Makes a call to the robot to move according to it's current position and the desired position 
        according to the input path and the current time. his method should call
        get_joint_positions and get_joint_velocities from the utils package to get the current joint 
        position and velocity and self._limb.set_joint_velocities() to set the joint velocity to something.  
        You may find joint_array_to_dict() in utils.py useful as well.

        Parameters
        ----------
        target_position: 7x' :obj:`numpy.ndarray` of desired positions
        target_velocity: 7x' :obj:`numpy.ndarray` of desired velocities
        target_acceleration: 7x' :obj:`numpy.ndarray` of desired accelerations
        """
        current_position = get_joint_positions(self._limb)
        current_velocity = get_joint_velocities(self._limb)
                                                
        err = target_position - current_position
        d_err = target_velocity - current_velocity

        Pout = self.Kp @ err
        Dout = self.Kv @ d_err

        controller_velocity = target_velocity + Pout + Dout
        # print(controller_velocity)

        self._limb.set_joint_velocities(joint_array_to_dict(controller_velocity, self._limb))

        # raise NotImplementedError
        # control_input = None
        # self._limb.set_joint_velocities(joint_array_to_dict(control_input, self._limb))

class PDJointTorqueController(Controller):
    def __init__(self, limb, kin, Kp, Kv):
        """
        Parameters
        ----------
        limb : :obj:`baxter_interface.Limb`
        kin : :obj:`BaxterKinematics`
        Kp : 7x' :obj:`numpy.ndarray`
        Kv : 7x' :obj:`numpy.ndarray`
        """
        Controller.__init__(self, limb, kin)
        self.Kp = np.diag(Kp)
        self.Kv = np.diag(Kv)
        self.is_jointspace_controller = True

    def step_control(self, target_position, target_velocity, target_acceleration):
        """
        Makes a call to the robot to move according to its current position and the desired position 
        according to the input path and the current time. This method should call
        get_joint_positions and get_joint_velocities from the utils package to get the current joint 
        position and velocity and self._limb.set_joint_torques() to set the joint torques to something. 
        You may find joint_array_to_dict() in utils.py useful as well.
        Recall that in order to implement a torque based controller you will need access to the 
        dynamics matrices M, C, G such that
        M ddq + C dq + G = u
        For this project, you will access the inertia matrix and gravity vector as follows:
        Inertia matrix: self._kin.inertia(positions_dict)
        Coriolis matrix: self._kin.coriolis(positions_dict, velocity_dict)
        Gravity matrix: self._kin.gravity(positions_dict) (You might want to scale this matrix by 0.01 or another scalar)
        These matrices were computed by a library and the coriolis matrix is approximate, 
        so you should think about what this means for the kinds of trajectories this 
        controller will be able to successfully track.
        Look in section 4.5 of MLS.
        Parameters
        ----------
        target_position: 7x' :obj:`numpy.ndarray` of desired positions
        target_velocity: 7x' :obj:`numpy.ndarray` of desired velocities
        target_acceleration: 7x' :obj:`numpy.ndarray` of desired accelerations
        """
        current_position = get_joint_positions(self._limb)
        current_velocity = get_joint_velocities(self._limb)
        
        current_position_dict = joint_array_to_dict(current_position, self._limb)
        current_velocity_dict = joint_array_to_dict(current_velocity, self._limb)
        inertia = self._kin.inertia(current_position_dict)
        coriolis = self._kin.coriolis(current_position_dict, current_velocity_dict)
        gravity = 0.01 * self._kin.gravity(current_position_dict) # (You might want to scale this matrix by 0.01 or another scalar)

                                                
        err = target_position - current_position
        d_err = target_velocity - current_velocity

        Pout = self.Kp @ err
        Dout = self.Kv @ d_err

        # a = (inertia.T @ (target_acceleration - Dout - Pout)).reshape(-1)
        # a = inertia @ (target_acceleration - Dout - Pout)
        M = inertia @ target_acceleration.T
        C = coriolis.T * current_velocity
        N = gravity
        feedback = Dout + Pout

        print("-----------------------------")
        sumval = np.array(M + C + N + feedback)

        controller_torque = np.array([i for i in sumval[0]])

        # controller_torque = np.array([0,0,0,0,0,1,0]) # Seems like the robot automatically accounts for gravity

        print(controller_torque, feedback)
        self._limb.set_joint_torques(joint_array_to_dict(controller_torque, self._limb))
