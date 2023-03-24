import random
import math
import os

import numpy as np

import PIL

from soccer_field import Field
from utils import create_scene, plot_robot, get_obs

def make_dataset(output_directory, num_points, seed, out_of_bounds=200):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    alphas = np.array([0.05**2, 0.005**2, 0.1**2, 0.01**2])
    beta = np.diag([np.deg2rad(5)**2])
    
    env = Field(alphas, beta)
    create_scene(env)
    robot_id = plot_robot(env,[50,50,0],[0])
    
    x_min = env.MARKER_OFFSET_X - out_of_bounds
    x_max = env.MARKER_OFFSET_X + env.MARKER_DIST_X + out_of_bounds
    y_min = env.MARKER_OFFSET_Y - out_of_bounds
    y_max = env.MARKER_OFFSET_Y + env.MARKER_DIST_Y + out_of_bounds
    
    for i in range(num_points):
        x = (random.random() * (x_max - x_min) + x_min)/100.
        y = (random.random() * (y_max - y_min) + y_min)/100.
        theta = random.random() * math.pi * 2.
        q = env.p.getQuaternionFromEuler([0,0,theta])
        env.p.resetBasePositionAndOrientation(robot_id, [x,y,0], q)
        rgbs, _, _ = get_obs(env, robot_id, resolution=32)
        f,b,l,r = rgbs
        f = f[:,:,:3]
        b = b[:,:,:3]
        l = l[:,:,:3]
        r = r[:,:,:3]
        rgb_strip = np.concatenate([l,f,r,b], axis=1)
        rgb_strip = np.concatenate(
            [rgb_strip[:,-16:], rgb_strip[:,:-16]], axis=1)
        
        image_name = '%s/rgb_%06i.jpg'%(output_directory, i)
        image = PIL.Image.fromarray(rgb_strip)
        image.save(image_name)
        print('Saved: %s'%image_name)
        
        label_name = '%s/label_%06i.npy'%(output_directory, i)
        label = np.array([x, y, theta])
        np.save(label_name, label)

if __name__ == '__main__':
    #make_dataset('train_dataset', 10000, seed=1)
    make_dataset('test_dataset', 1000, seed=2)
