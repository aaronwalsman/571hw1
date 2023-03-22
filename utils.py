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



def get_obs(env, car_id):
    car_pos, car_orient = env.p.getBasePositionAndOrientation(car_id)
    steering = env.p.getEulerFromQuaternion(car_orient)[2]

    # front camera
    front_cam = np.array([car_pos[0] + np.cos(steering) * 0.2, car_pos[1] + np.sin(steering) * 0.2, car_pos[2]+0.2])
    front_cam_to = np.array([car_pos[0] + np.cos(steering ) * 10, car_pos[1] + np.sin(steering) * 10, car_pos[2]+0.2])

    # back camera
    back_cam = np.array(car_pos)
    back_cam_to = np.array([car_pos[0] + np.cos(steering+np.pi) * 10, car_pos[1] + np.sin(steering+np.pi) * 10, car_pos[2]+0.2])


    # left camera
    left_cam = np.array([car_pos[0] + np.cos(steering) * 0.1, car_pos[1] + np.sin(steering) * 0.1, car_pos[2]+0.2])
    left_cam_to = np.array(
        [car_pos[0] + np.cos(steering + np.pi/2) * 10, car_pos[1] + np.sin(steering + np.pi/2) * 10, car_pos[2]+0.2])

    # right camera
    right_cam =  np.array([car_pos[0] + np.cos(steering) * 0.1, car_pos[1] + np.sin(steering) * 0.1, car_pos[2]+0.2])
    right_cam_to = np.array(
        [car_pos[0] + np.cos(steering - np.pi / 2) * 10, car_pos[1] + np.sin(steering - np.pi / 2) * 10,
         car_pos[2] + 0.2])



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
            fov=45,
            aspect=1.0,
            nearVal=0.1,
            farVal=100.0
        )
        # Add the camera to the scene
        _,_,rgb,depth,segm = env.p.getCameraImage(
            width = 512,
            height = 512,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=env.p.ER_BULLET_HARDWARE_OPENGL
        )


        images.append(rgb)
        depths.append(depth)
        masks.append(segm)

    return images, depths, masks


def plot_robot(env, x, z, radius=5):
    """Plot the robot on the soccer field."""
    print(x)
    racer_car_id = env.p.loadURDF("racecar.urdf", [x[0]/100, x[1]/100, 0], env.p.getQuaternionFromEuler([0,0,x[2]]))
    ray_from = [x[0]/100, x[1]/100, 0.05]
    ray_to = [x[0]/100 + np.cos(x[2]) * (radius + 0.1), x[1]/100+np.sin(x[2]) * (radius + 0.1), 0.05]
    # robot orientation
    env.p.addUserDebugLine(ray_from, ray_to, [0,0,0], 8)

    # observation
    ray_from = [x[0] / 100, x[1] / 100, 0.05]
    ray_to = [x[0] / 100 + np.cos(x[2]+z[0])*5, x[1] / 100 + np.sin(x[2]+z[0])*5, 0.05]
    env.p.addUserDebugLine(ray_from, ray_to, [0,0,1], 8)

    return racer_car_id


def plot_path(env, states, color):
    """Plot a path of states."""
    # ax = env.get_figure().gca()
    # ax.plot(states[:, 0], states[:, 1], color=color, linewidth=linewidth)
    prev = [states[0][0]/100, states[0][1]/100, 0.05]

    for state in states:
        rescaled_state = [state[0]/100, state[1]/100, 0.05]
        env.p.addUserDebugLine(prev, rescaled_state, color, 10)
        prev = rescaled_state



def create_scene(env):
    planeId = env.p.loadURDF("plane.urdf")
    h = 1
    r = 0.2

    pillar_shape = env.p.createCollisionShape(env.p.GEOM_CYLINDER, radius=r, height=h)
    # plot_particles(env, planeId)
    colors = [[0.9, 0, 0, 1], [0, 0.9, 0, 1], [0,0,0.9, 1],[0.5, 0.5, 0,1], [0, 0.5, 0.5, 1], [0.5, 0, 0.5, 1]]
    for m in env.MARKERS:
        x, y = env.MARKER_X_POS[m]/100, env.MARKER_Y_POS[m]/100
        pillar = env.p.createMultiBody(baseCollisionShapeIndex=pillar_shape,
                                   basePosition=[x, y, h/2])

        text_id = env.p.addUserDebugText(str(m),
                                     textPosition=[0,0,h/2+0.1],
                                     textColorRGB=[1, 0, 0],
                                     textSize=5,
                                     parentObjectUniqueId=pillar)
        env.p.changeVisualShape(pillar, -1, rgbaColor=colors[m-1])


def plot_particles(env, particles):
    # ax = env.get_figure().gca()
    # ax.scatter(particles[:, 0], particles[:, 1], s=0.5, c='k', marker='.')
    for each_particle in particles:
        env.p.addUserDebugPoints([each_particle[0], each_particle[1], 0.2], [1,0,0], 2)

