#!/usr/bin/env python -W ignore::DeprecationWarning
"""
Starter code for EE106B grasp planning project.
Author: Amay Saxena, Tiffany Cappellari
Modified by: Kirthi Kumar
"""
# may need more imports
import numpy as np
from numpy import linalg
from utils import *
import math
import trimesh
import vedo
import rospy
# from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest, GetPositionIKResponse
from geometry_msgs.msg import PoseStamped
# from moveit_commander import MoveGroupCommander

import sys

MAX_GRIPPER_DIST = 0.075
MIN_GRIPPER_DIST = 0.03
GRIPPER_LENGTH = 0.105
g = 9.81

import cvxpy as cvx # suggested, but you may change your solver to anything you'd like (ex. casadi)

def compute_force_closure(vertices, normals, num_facets, mu, gamma, object_mass):
    """
    Compute the force closure of some object at contacts, with normal vectors
    stored in normals. Since this is two contact grasp, we are using a basic algorithm
    wherein a grasp is in force closure as long as the line connecting the two contact
    points lies in both friction cones.

    Parameters
    ----------
    vertices (2x3 np.ndarray): obj mesh vertices on which the fingers will be placed
    normals (2x3 np.ndarray): obj mesh normals at the contact points
    num_facets (int): number of vectors to use to approximate the friction cone, vectors
        will be along the friction cone boundary
    mu (float): coefficient of friction
    gamma (float): torsional friction coefficient
    object_mass (float): mass of the object

    Returns
    -------
    (float): quality of the grasp
    """
    # Maximum antipodal line angle
    max_antipodal_angle = np.arctan(mu)
    normal1 = normals[0] ; normal2 = normals[1]
    vertex1 = vertices[0] ; vertex2 = vertices[1]
    antipodal_line = vertex1 - vertex2
    # Normalize vectors
    antipodal_line = antipodal_line / np.linalg.norm(antipodal_line)
    inward_n1 = -normal1 / np.linalg.norm(normal1)
    inward_n2 = -normal2 / np.linalg.norm(normal2)
    n1_angle = np.arccos(np.dot(-antipodal_line, inward_n1))
    n2_angle = np.arccos(np.dot(antipodal_line, inward_n2))
    return int(n1_angle < max_antipodal_angle and n2_angle < max_antipodal_angle and gamma > 0)

def get_grasp_map(vertices, normals, num_facets, mu, gamma):
    """
    Defined in the book on page 219. Compute the grasp map given the contact
    points and their surface normals

    Parameters
    ----------
    vertices (2x3 np.ndarray): obj mesh vertices on which the fingers will be placed
    normals (2x3 np.ndarray): obj mesh normals at the contact points
    num_facets (int): number of vectors to use to approximate the friction cone, vectors
        will be along the friction cone boundary
    mu (float): coefficient of friction
    gamma (float): torsional friction coefficient

    Returns
    -------
    (np.ndarray): grasp map
    """
    Bci = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]) # soft contact model
    vertex1 = vertices[0] ; vertex2 = vertices[1]
    normal1 = normals[0] ; normal2 = normals[1]
    x1 = np.cross(vertex1, np.array([0,0,1]))
    z1 = np.cross(x1, vertex1)
    x2 = np.cross(vertex2, np.array([0,0,1]))
    z2 = np.cross(x2, vertex2)
    R1 = np.vstack((x1, normal1, z1)).T
    R2 = np.vstack((x2, normal2, z2)).T
    g1_inv = np.vstack((np.hstack((R1.T, np.array([-R1.T @ vertex1]).T)), np.array([0,0,0,1])))
    g2_inv = np.vstack((np.hstack((R2.T, np.array([-R2.T @ vertex2]).T)), np.array([0,0,0,1])))
    G1 = adj(g1_inv).T @ Bci
    G2 = adj(g2_inv).T @ Bci
    return np.hstack((G1, G2))

def contact_forces_exist(vertices, normals, num_facets, mu, gamma, desired_wrench):
    """
    Compute whether the given grasp (at contacts with surface normals) can produce
    the desired_wrench. Will be used for gravity resistance.

    Parameters
    ----------
    vertices (2x3 np.ndarray): obj mesh vertices on which the fingers will be placed
    normals (2x3 np.ndarray): obj mesh normals at the contact points
    num_facets (int): number of vectors to use to approximate the friction cone, vectors
        will be along the friction cone boundary
    mu (float): coefficient of friction
    gamma (float): torsional friction coefficient
    desired_wrench (np.ndarray):potential wrench to be produced

    Returns
    -------
    (bool): whether contact forces can produce the desired_wrench on the object
    """
    Bci = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]) # soft contact model
    G = get_grasp_map(vertices, normals, num_facets, mu, gamma)
    F = np.array([ # Edges of the discretized friction cone
        [np.cos(i * 2 * np.pi / num_facets) for i in range(num_facets)],
        [np.sin(i * 2 * np.pi / num_facets) for i in range(num_facets)]
    ])  # aka F in section 1.1 (F * alpha = f)
    # Coefficients of the sum of the edges
    alpha1 = cvx.Variable(num_facets)
    alpha2 = cvx.Variable(num_facets)
    f1 = cvx.Variable(4)
    f2 = cvx.Variable(4)
    _f1 = Bci @ f1
    _f2 = Bci @ f2
    f = cvx.hstack([f1, f2]).T
    # Constraints
    constraints = [
        G @ f == desired_wrench, # f produces the desired wrench
        alpha1 >= 0, alpha2 >= 0, # negative friction does not exist
        f1[0:1] == F @ alpha1, f2[0:1] == F @ alpha2, # f is a linear combination of the edges
        # f1[0]**2 + f1[1]**2 <= mu**2 * f1[2]**2, f2[0]**2 + f2[1]**2 <= mu**2 * f2[2]**2, # f is in the friction cone
        cvx.abs(f1[3]) <= gamma * f1[2], cvx.abs(f2[3]) <= gamma * f2[2],
        # f1[2] > 0, f2[2] > 0 # normal force is positive
    ]
    objective = cvx.Minimize(cvx.norm(G @ f - desired_wrench, 2))
    prob = cvx.Problem(objective, constraints)
    result = prob.solve()
    if f.value is None:
        return result, 0, 0
    return result, 0, f.value
    

def compute_gravity_resistance(vertices, normals, num_facets, mu, gamma, object_mass):
    """
    Gravity produces some wrench on your object. Computes whether the grasp can
    produce an equal and opposite wrench.

    Parameters
    ----------
    vertices (2x3 np.ndarray): obj mesh vertices on which the fingers will be placed
    normals (2x3 np.ndarray): obj mesh normals at the contact points
    num_facets (int): number of vectors to use to approximate the friction cone, vectors
        will be along the friction cone boundary
    mu (float): coefficient of friction
    gamma (float): torsional friction coefficient
        torsional friction coefficient
    object_mass (float): mass of the object

    Returns
    -------
    (float): quality of the grasp
    """
    # Quality is determined as the minimum finger force required to counteract gravity
    grav_wrench = np.array([0, 0, -g * object_mass, 0, 0, 0]).T
    exists, _, _ = contact_forces_exist(vertices, normals, num_facets, mu, gamma, -grav_wrench)
    return exists

    # if contact_forces_exist doesn't work
    G = get_grasp_map(vertices, normals, num_facets, mu, gamma)
    f = cvx.Variable(6)
    constraints = [f >= 0, G @ f == grav_wrench]
    objective = cvx.Minimize(cvx.norm(f, 1))
    prob = cvx.Problem(objective, constraints)
    result = prob.solve()
    contact_force = result.value
    perpendicular_contact_force = np.linalg.norm(np.dot(contact_force, normals[0]) * normals[0])
    return perpendicular_contact_force

"""
you're encouraged to implement a version of this method,
def sample_around_vertices(delta, vertices, object_mesh=None):
    raise NotImplementedError
"""

def sample_around_vertices(mesh, vertices, normals, metric):
    if metric == "grav":
        perturb_magnitude = 0.005
        antinorm_magnitude = 0.05
    elif metric == "rfc":
        # perturb_magnitude = 0.005
        # antinorm_magnitude = 0.025
        perturb_magnitude = 0.001
        antinorm_magnitude = 0.015
    else:
        perturb_magnitude = 0.005
        antinorm_magnitude = 0.005
    vertex1 = vertices[0] ; vertex2 = vertices[1]
    normal1 = normals[0] ; normal2 = normals[1]
    x1 = np.cross(normal1, np.array([0,0,1]))
    z1 = np.cross(x1, normal1)
    p1_rot = np.vstack((x1, normal1, z1)).T
    x2 = np.cross(normal2, np.array([0,0,1]))
    z2 = np.cross(x2, normal2)
    p2_rot = np.vstack((x2, normal2, z2)).T
    p1_rand_angle = np.random.uniform(0, 2 * np.pi)
    p1_rand_mag = np.random.uniform(0, perturb_magnitude)
    p1_rand_dir = np.array([np.cos(p1_rand_angle), np.sin(p1_rand_angle), 0])
    random_p1 = vertex1 + antinorm_magnitude * normal1 + p1_rand_mag * p1_rot @ p1_rand_dir
    p2_rand_angle = np.random.uniform(0, 2 * np.pi)
    p2_rand_mag = np.random.uniform(0, perturb_magnitude)
    p2_rand_dir = np.array([np.cos(p2_rand_angle), np.sin(p2_rand_angle), 0])
    random_p2 = vertex2 + antinorm_magnitude * normal2 + p2_rand_mag * p2_rot @ p2_rand_dir
    # Find the points and corresponding normals on the object the random points would close on
    locations, face_ind = find_grasp_vertices(mesh, random_p1, random_p2)
    p1 = random_p1 ; p2 = random_p2
    # visualize_grasp(mesh, [p1, p2], get_gripper_pose([p1, p2], mesh))
    p1 = locations[0] ; p2 = locations[1]
    n1 = normal_at_point(mesh, p1) ; n2 = normal_at_point(mesh, p2)
    return [p1, p2], [n1, n2]

def compute_robust_force_closure(vertices, normals, mesh, num_facets, mu, gamma, object_mass):
    """
    Should return a score for the grasp according to the robust force closure metric.

    Parameters
    ----------
    vertices (2x3 np.ndarray): obj mesh vertices on which the fingers will be placed
    normals (2x3 np.ndarray): obj mesh normals at the contact points
    num_facets (int): number of vectors to use to approximate the friction cone, vectors
        will be along the friction cone boundary
    mu (float): coefficient of friction
    gamma (float): torsional friction coefficient
        torsional friction coefficient
    object_mass (float): mass of the object

    Returns
    -------
    (float): quality of the grasp
    """

    trialCount = 10
    closureSum = 0
    for i in range(trialCount):
        new_vertices, new_normals = sample_around_vertices(mesh, vertices, normals, "rfc")
        if new_vertices is None:
            continue
        # Compute the force closure of the random points
        closureSum += compute_force_closure(new_vertices, new_normals, num_facets, mu, gamma, object_mass)
    return closureSum / trialCount

def compute_ferrari_canny(vertices, normals, num_facets, mu, gamma, object_mass):
    """
    Should return a score for the grasp according to the Ferrari Canny metric.
    Use your favourite python convex optimization package. We suggest cvxpy.

    Parameters
    ----------
    vertices (2x3 np.ndarray): obj mesh vertices on which the fingers will be placed
    normals (2x3 np.ndarray): obj mesh normals at the contact points
    num_facets (int): number of vectors to use to approximate the friction cone, vectors
        will be along the friction cone boundary
    mu (float): coefficient of friction
    gamma (float): torsional friction coefficient
        torsional friction coefficient
    object_mass (float): mass of the object

    Returns
    -------
    (float): quality of the grasp
    """
    # Generate points on the unit sphere
    N = 1000 # point count
    points = np.zeros((N, 6))
    for i in range(N):
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        points[i] = np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi), 0, 0, 0])

    # find desired wrench
    desired_wrench = np.array([0, 0, -g * object_mass, 0, 0, 0]).T # using gravity wrench

    LQ = []
    trialCount = 1
    for i in range(trialCount):
        noise = points[np.random.randint(0, N)] # sample from set of random points
        wrench = desired_wrench #  + 0.001 * noise # add small perturbation
        wrench /= np.linalg.norm(wrench) # normalize wrench
        exists, _, arg_min = contact_forces_exist(vertices, normals, num_facets, mu, gamma, -desired_wrench) # check if contact forces exist
        if exists:
            LQ.append(np.linalg.norm(arg_min)**2) # l2norm^2 of f_optimal

    LQ = 1 / np.sqrt(np.array(LQ))
    # print(LQ)
    return min(LQ)


def custom_grasp_planner(object_mesh, vertices, metric, grasp_set):
    """
    Write your own grasp planning algorithm! You will take as input the mesh
    of an object, and a pair of contact points from the surface of the mesh.
    You should return a 4x4 rigid transform specifying the desired pose of the
    end-effector (the gripper tip) that you would like the gripper to be at
    before closing in order to execute your grasp.

    You should be prepared to handle malformed grasps. Return None if no
    good grasp is possible with the provided pair of contact points.
    Keep in mind the constraints of the gripper (length, minimum and maximum
    distance between fingers, etc) when picking a good pose, and also keep in
    mind limitations of the robot (can the robot approach a grasp from the inside
    of the mesh? How about from below?). You should also make sure that the robot
    can successfully make contact with the given contact points without colliding
    with the mesh.

    The trimesh package has several useful functions that allow you to check for
    collisions between meshes and rays, between meshes and other meshes, etc, which
    you may want to use to make sure your grasp is not in collision with the mesh.

    Take a look at the functions find_intersections, find_grasp_vertices,
    normal_at_point in utils.py for examples of how you might use these trimesh
    utilities. Be wary of using these functions directly. While they will probably
    work, they don't do excessive edge-case handling. You should spend some time
    reading the documentation of these packages to find other useful utilities.
    You may also find the collision, proximity, and intersections modules of trimesh
    useful.

    Feel free to change the signature of this function to add more arguments
    if you believe they will be useful to your planner.

    Parameters
    ----------
    object_mesh (trimesh.base.Trimesh): A triangular mesh of the object, as loaded in with trimesh.
    vertices (2x3 np.ndarray): obj mesh vertices on which the fingers will be placed

    Returns
    -------
    (4x4 np.ndarray): The rigid transform for the desired pose of the gripper, in the object's reference frame.
    """

    # constants -- you may or may not want to use these variables below
    num_facets = 64
    mu = 0.5
    gamma = 0.1
    object_mass = 0.25
    g = 9.8
    desired_wrench = np.array([0, 0, g * object_mass, 0, 0, 0]).T

    trials = 100
    delta = 0.04
    gravity_th = 3
    ferrari_th = 1
    force_c_th = 0.0

    poses = []
    print(metric + " Metric")

    for trial in range(trials):
        # print("Grasp set ", grasp_set)
        normals = np.array([
            normal_at_point(object_mesh, vertices[0]),
            normal_at_point(object_mesh, vertices[1])
        ])
        test_vert, test_norm = sample_around_vertices(object_mesh, vertices, normals, metric)
        
        # if compute_force_closure(test_vert, test_norm, num_facets, mu, gamma, object_mass) != 0:
        if metric == "grav":
            metric_val = compute_gravity_resistance(test_vert, test_norm, num_facets, mu, gamma, object_mass)
        elif metric == "ferrari":
            metric_val = compute_ferrari_canny(test_vert, test_norm, num_facets, mu, gamma, object_mass)
        elif metric == "rfc":
            metric_val = compute_robust_force_closure(test_vert, test_norm, object_mesh, num_facets, mu, gamma, object_mass)
        else:
            raise ValueError("Invalid metric")
        # print(metric_val)
        poses.append({
            "pose": get_gripper_pose(test_vert, object_mesh),
            "vertices": test_vert,
            "metric": metric_val
        })

    if len(poses) == 0:
        return None # return None if malformed grasps, no good grasp found

    def comp(p):
        if p['metric'] == np.inf:
            return -np.inf
        return p['metric']

    # Sort the poses by metric
    return max(poses, key=comp)
                
def get_gripper_pose(vertices, object_mesh): # you may or may not need this method
    """
    Creates a 3D Rotation Matrix at the origin such that the y axis is the same
    as the direction specified.  There are infinitely many of such matrices,
    but we choose the one where the z axis is as vertical as possible.
    z -> y
    x -> x
    y -> z

    Parameters
    ----------
    origin : 3x1 :obj:`numpy.ndarray`
    x : 3x1 :obj:`numpy.ndarray`

    Returns
    -------
    4x4 :obj:`numpy.ndarray`
    """
    origin = np.mean(vertices, axis=0)
    direction = vertices[0] - vertices[1]

    up = np.array([0, 0, 1])
    y = normalize(direction)
    x = normalize(np.cross(up, y))
    z = np.cross(x, y)

    gripper_top = origin + GRIPPER_LENGTH * z
    gripper_double = origin + 2 * GRIPPER_LENGTH * z
    if len(find_intersections(object_mesh, gripper_top, gripper_double)[0]) > 0:
        z = normalize(np.cross(up, y))
        x = np.cross(y, x)
    result = np.eye(4)
    result[0:3,0] = x
    result[0:3,1] = y
    result[0:3,2] = z
    result[0:3,3] = origin
    return result


def visualize_grasp(mesh, vertices, pose):
    """Visualizes a grasp on an object. Object specified by a mesh, as
    loaded by trimesh. vertices is a pair of (x, y, z) contact points.
    pose is the pose of the gripper tip.

    Parameters
    ----------
    mesh (trimesh.base.Trimesh): mesh of the object
    vertices (np.ndarray): 2x3 matrix, coordinates of the 2 contact points
    pose (np.ndarray): 4x4 homogenous transform matrix
    """
    p1, p2 = vertices
    center = (p1 + p2) / 2
    approach = pose[:3, 2]
    tail = center - GRIPPER_LENGTH * approach

    contact_points = []
    for v in vertices:
        contact_points.append(vedo.Point(pos=v, r=30))

    vec = (p1 - p2) / np.linalg.norm(p1 - p2)
    line = vedo.shapes.Tube([center + 0.5 * MAX_GRIPPER_DIST * vec,
                                   center - 0.5 * MAX_GRIPPER_DIST * vec], r=0.001, c='g')
    approach = vedo.shapes.Tube([center, tail], r=0.001, c='g')
    vedo.show([mesh, line, approach] + contact_points, new=True)


def randomly_sample_from_mesh(mesh, n):
    """Example of sampling points from the surface of a mesh.
    Returns n (x, y, z) points sampled from the surface of the input mesh
    uniformly at random. Also returns the corresponding surface normals.

    Parameters
    ----------
    mesh (trimesh.base.Trimesh): mesh of the object
    n (int): number of desired sample points

    Returns
    -------
    vertices (np.ndarray): nx3 matrix, coordinates of the n surface points
    normals (np.ndarray): nx3 matrix, normals of the n surface points
    """
    vertices, face_ind = trimesh.sample.sample_surface(mesh, 3) # you may want to check out the trimesh mehtods here:)
    normals = mesh.face_normals[face_ind]
    return vertices, normals


def load_grasp_data(object_name):
    """Loads grasp data from the provided NPZ files. It returns three arrays:

    Parameters
    ----------
    object_name (String): type of object

    Returns
    -------
    vertices (np.ndarray): nx3 matrix, coordinates of the n surface points
    normals (np.ndarray): nx3 matrix, normals of the n surface points

    grasp_vertices (np.ndarray): 5x2x3 matrix. For each of the 5 grasps,
            this stores a pair of (x, y, z) locations that are the contact points
            of the grasp.
    normals (np.ndarray): 5x2x3 matrix. For each grasp, stores the normal
            vector to the mesh at the two contact points. Remember that the normal
            vectors to a closed mesh always point OUTWARD from the mesh.
    tip_poses (np.ndarray): 5x4x4 matrix. For each of the five grasps, this
            stores the 4x4 rigid transform of the reference frame of the gripper
            tip before the gripper is closed in order to grasp the object.
    results (np.ndarray): 5x5 matrix. Stores the result of five trials for
            each of the five grasps. Entry (i, j) is a 1 if the jth trial of the
            ith grasp was successful, 0 otherwise.
    """
    data = np.load('src/proj4_pkg/grasp_data/{}.npz'.format(object_name))
    return data['grasp_vertices'], data['normals'], data['tip_poses'], data['results']


def load_mesh(object_name):
    mesh = trimesh.load_mesh("src/proj4_pkg/objects/{}.obj".format(object_name))
    mesh.fix_normals()
    return mesh

# def get_gripper(obj="pawn", metric="grav"):
#     # obj should be 'pawn' or 'nozzle'.
#     vertices, normals, poses, results = load_grasp_data(obj)
#     mesh = load_mesh(obj)
#     # for v, p in zip(vertices, poses):
#         # visualize_grasp(mesh, v, p)
        
#     poses = []
#     for i in range(len(vertices)):
#         poses.append(custom_grasp_planner(mesh, vertices[i], metric, i))
#     # poses = [custom_grasp_planner(mesh, v, metric) for v in vertices]
#     print(poses)
#     return poses[0]['pose']

def main():
    """ Example for interacting with the codebase. Loads data and
    visualizes each grasp against the corresponding mesh of the given
    object.
    """
    # obj = 'pawn' # should be 'pawn' or 'nozzle'.
    obj = 'pawn'
    vertices, normals, poses, results = load_grasp_data(obj)
    mesh = load_mesh(obj)
    # for v, p in zip(vertices, poses):
    #     visualize_grasp(mesh, v, p)
        
    # poses = [custom_grasp_planner(mesh, v, "rfc") for v in vertices]
    poses = []
    for i in range(len(vertices)):
        if i in []:
            print("skip")
        else:
            # poses.append(custom_grasp_planner(mesh, vertices[i], "grav", i))
            # poses.append(custom_grasp_planner(mesh, vertices[i], "rfc", i))
            poses.append(custom_grasp_planner(mesh, vertices[i], "ferrari", i))
    print(poses)
    for i in range(len(vertices)):
        visualize_grasp(mesh, poses[i]["vertices"], poses[i]["pose"])

    
if __name__ == '__main__':
    main()
