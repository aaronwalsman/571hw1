""" Written by Brian Hou for CSE571: Probabilistic Robotics
"""

import numpy as np

from utils import minimized_angle


class ExtendedKalmanFilter:
    def __init__(self, mean, cov, alphas, beta):
        self.alphas = alphas
        self.beta = beta

        self._init_mean = mean
        self._init_cov = cov
        self.reset()

    def reset(self):
        self.mu = self._init_mean
        self.sigma = self._init_cov

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving a landmark
        observation.

        u: action
        z: landmark observation
        marker_id: landmark ID
        """
        # YOUR IMPLEMENTATION HERE
        # PREDICTION #
        mu_pred, G_t = env.G(self.mu, u)
        V_t = env.V(self.mu, u)
        cov_pred = G_t @ self.sigma @ G_t.T + V_t @ env.noise_from_motion(u, self.alphas) @ V_t.T

        # CORRECTION
        H_t = env.H(mu_pred, marker_id)
        S_t = H_t @ cov_pred @ H_t.T + self.beta
        K_t = cov_pred @ H_t.T @ (1/S_t) # (3x3) @ (3x1) @ (1x1) = (3x1)
        self.mu = mu_pred.reshape(3,1) + K_t @ (z-env.observe(mu_pred, marker_id))
        self.sigma = (np.eye(3) - K_t @ H_t) @ cov_pred

        return self.mu, self.sigma
