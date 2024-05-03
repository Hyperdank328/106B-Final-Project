#!/usr/bin/env python

import numpy as np
import math
from utils.utils import *

def rotation_2d(theta):
    """
    Computes a 2D rotation matrix given the angle of rotation.
    
    Args:
    theta: the angle of rotation
    
    Returns:
    rot - (2,2) ndarray: the resulting rotation matrix
    """
    
    rot = np.zeros((2,2))
    rot[0,0] = np.cos(theta)
    rot[1,1] = np.cos(theta)
    rot[0,1] = -np.sin(theta)
    rot[1,0] = np.sin(theta)

    return rot

def hat_2d(xi):
    """
    Converts a 2D twist to its corresponding 3x3 matrix representation
    
    Args:
    xi - (3,) ndarray: the 2D twist
    
    Returns:
    xi_hat - (3,3) ndarray: the resulting 3x3 matrix
    """
    if not xi.shape == (3,):
        raise TypeError('omega must be a 3-vector')

    xi_hat = np.zeros((3,3))
    xi_hat[0,1] = -xi[2]
    xi_hat[1,0] =  xi[2]
    xi_hat[0:2,2] = xi[0:2]

    return xi_hat

def homog_2d(xi, theta):
    """
    Computes a 3x3 homogeneous transformation matrix given a 2D twist and a 
    joint displacement
    
    Args:
    xi - (3,) ndarray: the 2D twiprint("---------------")st
    theta: the joint displacement
    
    Returns:
    g - (3,3) ndarray: the resulting homogeneous transformation matrix
    """
    if not xi.shape == (3,):
        raise TypeError('xi must be a 3-vector')

    g = np.zeros((3,3))
    wtheta = xi[2]*theta
    R = rotation_2d(wtheta)
    p = np.dot(np.dot( \
        [[1 - np.cos(wtheta), np.sin(wtheta)],
        [-np.sin(wtheta), 1 - np.cos(wtheta)]], \
        [[0,-1],[1,0]]), \
        [[xi[0]/xi[2]],[xi[1]/xi[2]]])

    g[0:2,0:2] = R
    g[0:2,2:3] = p[0:2]
    g[2,2] = 1

    return g

#-----------------------------3D Functions--------------------------------------
#-------------(These are the functions you need to complete)--------------------

def skew_3d(omega):
    """
    Converts a rotation vector in 3D to its corresponding skew-symmetric matrix.
    
    Args:
    omega - (3,) ndarray: the rotation vector
    
    Returns:
    omega_hat - (3,3) ndarray: the corresponding skew symmetric matrix
    """

    o1 = omega[0]
    o2 = omega[1]
    o3 = omega[2]
    ans = np.array([
                    [0, -o3, o2],
                    [o3, 0, -o1],
                    [-o2, o1, 0]
                    ])
    return ans


def rotation_3d(omega, theta):
    """
    Computes a 3D rotation matrix given a rotation axis and angle of rotation.
    
    Args:
    omega - (3,) ndarray: the axis of rotation
    theta: the angle of rotation
    
    Returns:
    rot - (3,3) ndarray: the resulting rotation matrix
    """

    I = np.eye(3, dtype=float)
    wow = np.divide(skew_3d(omega), np.linalg.norm(omega))
    wowpow = np.divide(np.linalg.matrix_power(skew_3d(omega), 2), np.power(np.linalg.norm(omega), 2))
    normwth = np.linalg.norm(omega) * theta
    rotation = I + (np.multiply(wow, np.sin(normwth))) + np.multiply(wowpow, (1 - np.cos(normwth)))
    return rotation

def hat_3d(xi):
    """
    Converts a 3D twist to its corresponding 4x4 matrix representation
    
    Args:
    xi - (6,) ndarray: the 3D twist
    
    Returns:
    xi_hat - (4,4) ndarray: the corresponding 4x4 matrix
    """

    vx = xi[0]
    vy = xi[1]
    vz = xi[2]
    wx = xi[3]
    wy = xi[4]
    wz = xi[5]

    r = np.array([
        [0, -wz, wy, vx],
        [wz, 0, -wx, vy],
        [-wy, wx, 0, vz],
        [0,0,0,0]
    ])

    return r



def homog_3d(xi, theta):
    """
    Computes a 4x4 homogeneous transformation matrix given a 3D twist and a 
    joint displacement.
    
    Args:
    xi - (6,) ndarray: the 3D twist
    theta: the joint displacement
    Returns:
    g - (4,4) ndarary: the resulting homogeneous transformation matrix
    """

    v = np.array([xi[0], xi[1], xi[2]])
    w = np.array([xi[3], xi[4], xi[5]])
    wa = np.array([[xi[3], xi[4], xi[5]]])
    va = np.array([[xi[0], xi[1], xi[2]]])
    I = np.eye(3, dtype=float)




    if all([x == 0 for x in w]):
        ans = np.hstack((I, va.T))
        ans = np.vstack((ans, [0, 0, 0, 1]))
        return ans
    
    normsq = 1 / np.power(np.linalg.norm(w), 2)
    one_minus_rot = I - rotation_3d(w, theta)
    omega_v = skew_3d(w) @ v
    wwtvth = np.multiply((wa.T @ wa) @ v, theta)
    

    term = np.multiply(normsq, (one_minus_rot @ omega_v) + wwtvth)
    ta = np.array([[term[0], term[1], term[2]]])
    ans = np.hstack((rotation_3d(w, theta), ta.T))
    ans = np.vstack((ans, [0, 0, 0, 1]))
    return ans
def vee(hat):
    if hat.shape == (3, 3):
        return np.array([hat[2][1], hat[0][2], hat[1][0]])
    elif hat.shape == (4, 4):
        return np.array([hat[0, 3], hat[1, 3], hat[2, 3], hat[2, 1], hat[0, 2], hat[1, 0]])
    else:
        raise ValueError
def log(g):
    
    R = g[0:3, 0:3]
    p = g[0:3, 3]
    theta = np.arccos((np.trace(R) - 1) / 2)
    # print("-------")
    # print(R)
    omegahat = theta / (2 * np.sin(theta)) * (R - R.T)
    omega = vee(omegahat)
    norm = np.linalg.norm(omega)
    Ainv = np.eye(3) \
          - omegahat / 2 \
           + ((2 * np.sin(norm) - norm * (1 + np.cos(norm))) / (2 * norm ** 2 * np.sin(norm))) * omegahat @ omegahat
    # print(Ainv)
    # print(Ainv @ p)
    # print("------")
    return np.vstack((np.hstack((omegahat, (Ainv @ p).reshape((3,1)))), np.array([0, 0, 0, 0])))
def get_g_fk(kin, limb, joint_pos):

    vec = kin.forward_position_kinematics(joint_array_to_dict(joint_pos, limb))
    pos = vec[0:3]
    quat = vec[3:7]
    return get_g_matrix(pos, quat)
def get_g_vfk(kin, limb, joint_vel):

    twist = kin.forward_velocity_kinematics(joint_array_to_dict(joint_vel, limb))

    return np.array([twist.vel.x(), twist.vel.y(), twist.vel.z(), twist.rot.x(), twist.rot.y(), twist.rot.z()])
# xi = np.array([1, 2, 3, 1, 0, 0])
# exp = homog_3d(xi, 1)
# print(hat(xi))
# print(log(exp))

# print(hat(xi))
# print(vee(hat(xi)))
