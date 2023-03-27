""" Written by Brian Hou for CSE571: Probabilistic Robotics (Winter 2019)
    Modified by Wentao Yuan for CSE571: Probabilistic Robotics (Spring 2022)
    Modified by Aaron Walsman and Zoey Chen for CSEP590A: Robotics (Spring 2023)
"""

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    minimized_angle,
    create_scene,
    add_robot,
    move_robot,
    plot_observation,
    plot_path_step,
    plot_particles,
)
from soccer_field import Field
import policies
from ekf import ExtendedKalmanFilter
from pf import ParticleFilter


def localize(
    env,
    policy,
    filt,
    x0,
    num_steps,
    plot=False,
    step_pause=None,
    step_breakpoint=False
):
    # Collect data from an entire rollout
    (states_noisefree,
     states_real,
     action_noisefree,
     obs_noisefree,
     obs_real) = env.rollout(x0, policy, num_steps)
    states_filter = np.zeros(states_real.shape)
    states_filter[0, :] = x0.ravel()

    errors = np.zeros((num_steps, 3))
    position_errors = np.zeros(num_steps)
    mahalanobis_errors = np.zeros(num_steps)

    if plot:
        pillar_ids, text_ids = create_scene(env)
        car_id = add_robot(env)
        obs_id = None
        particle_id = None
    
    for i in range(num_steps):
        x_real = states_real[i+1, :].reshape((-1, 1))
        u_noisefree = action_noisefree[i, :].reshape((-1, 1))
        z_real = obs_real[i, :].reshape((-1, 1))
        marker_id = env.get_marker_id(i)

        if filt is None:
            mean, cov = x_real, np.eye(3)
        else:
            # filters only know the action and observation
            mean, cov = filt.update(env, u_noisefree, z_real, marker_id)
        states_filter[i+1, :] = mean.ravel()
        
        if plot:
            # move the robot
            move_robot(env, car_id, x_real)
            
            # plot observation
            obs_id = plot_observation(
                env, obs_id, x_real, z_real, marker_id, [0,0,0])
            
            # plot actual trajectory
            x_real_previous = states_real[i, :].reshape((-1, 1))
            plot_path_step(env, x_real_previous, x_real, [0,0,1])
            
            # plot noisefree trajectory
            noisefree_previous = states_noisefree[i]
            noisefree_current = states_noisefree[i+1]
            plot_path_step(env, noisefree_previous, noisefree_current, [0,1,0])
            
            # plot estimated trajectory
            if filt is not None:
                filter_previous = states_filter[i]
                filter_current = states_filter[i+1]
                plot_path_step(env, filter_previous, filter_current, [1,0,0])
            
            # plot particles
            if args.filter_type == 'pf':
                particle_id = plot_particles(
                    env,
                    filt.particles,
                    filt.weights,
                    previous_particles=particle_id
                )
        
        # pause/breakpoint
        if step_pause:
            time.sleep(step_pause)
        if step_breakpoint:
            breakpoint()
        
        errors[i, :] = (mean - x_real).ravel()
        errors[i, 2] = minimized_angle(errors[i, 2])
        position_errors[i] = np.linalg.norm(errors[i, :2])

        cond_number = np.linalg.cond(cov)
        if cond_number > 1e12:
            print('Badly conditioned cov (setting to identity):', cond_number)
            print(cov)
            cov = np.eye(3)
        mahalanobis_errors[i] = \
            errors[i:i+1, :].dot(np.linalg.inv(cov)).dot(errors[i:i+1, :].T)

    mean_position_error = position_errors.mean()
    mean_mahalanobis_error = mahalanobis_errors.mean()
    anees = mean_mahalanobis_error / 3

    if filt is not None:
        print('-' * 80)
        print('Mean position error:', mean_position_error)
        print('Mean Mahalanobis error:', mean_mahalanobis_error)
        print('ANEES:', anees)
    
    if plot:
        while True:
            env.p.stepSimulation()
    
    return mean_position_error, anees


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'filter_type', choices=('none', 'ekf', 'pf'),
        help='filter to use for localization')
    parser.add_argument(
        '--plot', action='store_true',
        help='turn on plotting')
    parser.add_argument(
        '--seed', type=int,
        help='random seed')
    parser.add_argument(
        '--num-steps', type=int, default=200,
        help='timesteps to simulate')

    # Noise scaling factors
    parser.add_argument(
        '--data-factor', type=float, default=1,
        help='scaling factor for motion and observation noise (data)')
    parser.add_argument(
        '--filter-factor', type=float, default=1,
        help='scaling factor for motion and observation noise (filter)')
    parser.add_argument(
        '--num-particles', type=int, default=100,
        help='number of particles (particle filter only)')
    
    # Debugging arguments
    parser.add_argument(
        '--step-pause', type=float, default=0.,
        help='slows down the rollout to make it easier to visualize')
    parser.add_argument(
        '--step-breakpoint', action='store_true',
        help='adds a breakpoint to each step for debugging purposes')
    
    return parser


if __name__ == '__main__':
    args = setup_parser().parse_args()
    print('Data factor:', args.data_factor)
    print('Filter factor:', args.filter_factor)

    if args.seed is not None:
        np.random.seed(args.seed)

    alphas = np.array([0.05**2, 0.005**2, 0.1**2, 0.01**2])
    beta = np.diag([np.deg2rad(5)**2])

    env = Field(
        args.data_factor * alphas,
        args.data_factor * beta,
        gui=args.plot,
    )
    policy = policies.OpenLoopRectanglePolicy()

    initial_mean = np.array([180, 50, 0]).reshape((-1, 1))
    initial_cov = np.diag([10, 10, 1])

    if args.filter_type == 'none':
        filt = None
    elif args.filter_type == 'ekf':
        filt = ExtendedKalmanFilter(
            initial_mean,
            initial_cov,
            args.filter_factor * alphas,
            args.filter_factor * beta
        )
    elif args.filter_type == 'pf':
        filt = ParticleFilter(
            initial_mean,
            initial_cov,
            args.num_particles,
            args.filter_factor * alphas,
            args.filter_factor * beta
        )

    # You may want to edit this line to run multiple localization experiments.
    localize(env, policy, filt, initial_mean, args.num_steps, args.plot, args.step_pause, args.step_breakpoint)
