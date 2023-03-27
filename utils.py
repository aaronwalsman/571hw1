""" Written by Brian Hou for CSE571: Probabilistic Robotics (Winter 2019)
    Modified by Zoey Chen for CSE590A: Probabilistic Robotics (Spring 2023)
"""

import numpy as np
import matplotlib.pyplot as plt
import math

def minimized_angle(angle):
    """Normalize an angle to [-pi, pi]."""
    while angle < -np.pi:
        angle += 2 * np.pi
    while angle >= np.pi:
        angle -= 2 * np.pi
    return angle

def get_obs(env, car_id, resolution=32):
    car_pos, car_orient = env.p.getBasePositionAndOrientation(car_id)
    steering = env.p.getEulerFromQuaternion(car_orient)[2]
    
    camera_height = 0.2
    
    # front camera
    front_cam = np.array(car_pos) + [0,0,camera_height]
    front_cam_to = np.array([
        car_pos[0] + np.cos(steering) * 10,
        car_pos[1] + np.sin(steering) * 10,
        car_pos[2] + camera_height,
    ])

    # back camera
    back_cam = np.array(car_pos) + [0,0,camera_height]
    back_cam_to = np.array([
        car_pos[0] + np.cos(steering+np.pi) * 10,
        car_pos[1] + np.sin(steering+np.pi) * 10,
        car_pos[2] + camera_height,
    ])

    # left camera
    left_cam = np.array(car_pos) + [0,0,camera_height]
    left_cam_to = np.array([
        car_pos[0] + np.cos(steering + np.pi/2) * 10,
        car_pos[1] + np.sin(steering + np.pi/2) * 10,
        car_pos[2] + camera_height,
    ])

    # right camera
    right_cam = np.array(car_pos) + [0,0,camera_height]
    right_cam_to = np.array([
        car_pos[0] + np.cos(steering - np.pi / 2) * 10,
        car_pos[1] + np.sin(steering - np.pi / 2) * 10,
        car_pos[2] + camera_height,
    ])

    cam_eyes = [front_cam, back_cam, left_cam, right_cam]
    cam_targets = [front_cam_to, back_cam_to, left_cam_to, right_cam_to]

    images = []
    depths = []
    masks = []
    for i in range(4):
        # Define the camera view matrix
        view_matrix = env.p.computeViewMatrix(
            cameraEyePosition=cam_eyes[i],
            cameraTargetPosition=cam_targets[i],
            cameraUpVector = [0,0,1]
        )
        # Define the camera projection matrix
        projection_matrix = env.p.computeProjectionMatrixFOV(
            fov=90,
            aspect=1.0,
            nearVal=0.1,
            farVal=100.0
        )
        # Add the camera to the scene
        _,_,rgb,depth,segm = env.p.getCameraImage(
            width = resolution,
            height = resolution,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=env.p.ER_BULLET_HARDWARE_OPENGL
        )

        images.append(rgb)
        depths.append(depth)
        masks.append(segm)

    return images, depths, masks

def add_robot(env): #, x, z): #, radius=5):
    """Add the robot to the environment."""
    racer_car_id = env.p.loadURDF("racecar.urdf", [0,0,0], [0,0,0,1])
    
    return racer_car_id

def move_robot(env, racer_car_id, x):
    p = [x[0]/100., x[1]/100., 0]
    q = env.p.getQuaternionFromEuler([0,0,x[2]+np.pi])
    env.p.resetBasePositionAndOrientation(racer_car_id, p, q)

def plot_observation(env, obs_id, x, z, marker_id, color):
    xyz0 = np.array([x[0,0]/100., x[1,0]/100., 0.05])
    
    marker_x = np.array([
        env.MARKER_X_POS[marker_id]/100.,
        env.MARKER_Y_POS[marker_id]/100.,
        0.05
    ])
    distance = np.linalg.norm(xyz0 - marker_x)
    
    dx = np.cos(z+x[2])
    dy = np.sin(z+x[2])
    xyz1 = [x[0]/100. + dx*distance, x[1]/100. + dy*distance, 0.05]
    
    kwargs = {}
    if obs_id is not None:
        kwargs['replaceItemUniqueId'] = obs_id
    obs_id = env.p.addUserDebugLine(
            xyz0, xyz1, color, 2, **kwargs)
    
    return obs_id

def plot_path_step(env, x_previous, x_current, color):
    xyz_previous = [x_previous[0]/100., x_previous[1]/100., 0.05]
    xyz_current = [x_current[0]/100., x_current[1]/100., 0.05]
    env.p.addUserDebugLine(xyz_previous, xyz_current, color, 2)

def create_scene(env):
    planeId = env.p.loadURDF("plane.urdf")
    h = 1
    r = 0.1

    pillar_shape = env.p.createCollisionShape(
        env.p.GEOM_CYLINDER, radius=r, height=h)
    colors = [
        [0.9, 0.0, 0.0, 1.0],
        [0.0, 0.9, 0.0, 1.0],
        [0.0, 0.0, 0.9, 1.0],
        [0.5, 0.5, 0.0 ,1.0],
        [0.0, 0.5, 0.5, 1.0],
        [0.5, 0.0, 0.5, 1.0],
    ]
    pillar_ids = []
    text_ids = []
    for m in env.MARKERS:
        x, y = env.MARKER_X_POS[m]/100, env.MARKER_Y_POS[m]/100
        pillar_id = env.p.createMultiBody(
            baseCollisionShapeIndex=pillar_shape, basePosition=[x, y, h/2])
        pillar_ids.append(pillar_id)
        env.p.setCollisionFilterGroupMask(pillar_id, -1, 0, 0)
        env.p.changeVisualShape(pillar_id, -1, rgbaColor=colors[m-1])
        
        text_id = env.p.addUserDebugText(
            str(m),
            textPosition=[0,0,h/2+0.1],
            textColorRGB=[0, 0, 0],
            textSize=2,
            parentObjectUniqueId=pillar_id,
        )
        text_ids.append(text_id)
    
    return pillar_ids, text_ids

def plot_particles(env, particles, weights, previous_particles=None):
    xyz = np.concatenate(
        (particles[:,:2]/100., np.full((len(particles),1), 0.2)), axis=1)
    color = np.zeros((len(particles),3))
    color[:,0] = 1
    color = color * weights.reshape(-1,1) * 50
    color = np.clip(color, 0, 1)
    kwargs = {}
    if previous_particles is not None:
        kwargs['replaceItemUniqueId'] = previous_particles
    particle_id = env.p.addUserDebugPoints(xyz, color, pointSize=2, **kwargs)
    
    return particle_id
