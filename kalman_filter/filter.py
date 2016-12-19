#!/usr/bin/env python3
"""
Demonstrate work of kalman filter
"""
from __future__ import print_function
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import norm
from numpy import matmul


def parse_args(*argument_array):
  parser = argparse.ArgumentParser()
  parser.add_argument('--step', type=int, default=100)
  args = parser.parse_args(*argument_array)
  return args


onezero_column = np.array([[1], [0]])
zeroone_column = np.array([[0], [1]])


class KalmanFilter:
  H = np.array([[1, 0]])

  def __init__(self, x_0):
    self.x = [x_0]
    print('x_0 shape', x_0.shape)
    self.post_x = x_0
    # estimate covariance
    self.post_p = np.identity(2)
    self.v_history = []
    self.vvT_history = []

  def step(self, obs, delta_t):
    print('step...')
    F = np.array([[1, delta_t], [0, 1]])
    # G = np.array([.5 * delta_t ** 2, delta_t])

    # if len(self.x_history) > 1:
    #   var_v = np.var([v * v for _, v in self.x_history])
    # if len(self.a_history) > 1:
    #   var_a = np.var([a * a for a in self.a_history])

    # Q = np.array([[0.001, 0.001],
    #               [0.001, 0.001]])
    G = np.array([[delta_t ** 2 / 2], [delta_t]])
    Q = 1e-6 * matmul(G, G.T) / (delta_t ** 2)
    R = delta_t ** 2 * np.array([[1.]])

    prior_x = matmul(F, self.post_x)
    prior_p = _sandwich(F, self.post_p) + Q

    obs_loc = np.array(obs)
    obs_v = obs_loc - self.post_x[0]
    obs_x = matmul(zeroone_column, [obs_v]) + matmul(onezero_column, [obs_loc])
    est_step_v = obs_v / delta_t
    self.vvT_history.append(matmul(est_step_v, est_step_v.T))
    self.v_history.append(est_step_v[0])
    # R = R * np.mean(self.vvT_history)

    measurement_residual = obs_loc - np.matmul(self.H, prior_x)
    covariance = _sandwich(self.H, prior_p) + R
    kalman_gain = np.matmul(np.matmul(prior_p, self.H.T),
                            inv(covariance))
    print('kalman gain', kalman_gain)
    post_x = prior_x + np.matmul(kalman_gain, measurement_residual)
    post_p = np.matmul(np.identity(2) - np.matmul(kalman_gain, self.H),
                       prior_p)
    self.post_x, self.post_p = post_x, post_p
    return np.matmul(self.H, post_x)


def _sandwich(A, B):
  return np.matmul(np.matmul(A, B), A.T)


def main(args):
  df = pd.read_csv('/Users/arsen/Workspace/macrobase/bench/workflows/moving_gaussian_nov29_grid/11-29-15_32_52/speed=0.00100.csv')
  N = df.shape[0]
  step = args.step
  ll = []
  for i in range(0, N, step):
    ll.append(np.mean(df[i:i + step]))
  print('var', np.var(ll))
  f = KalmanFilter(np.array([[0., 0.],
                             [0., 0.]]))
  # filtered = [f.step([l], step)[0] for l in ll]
  filtered = [f.step([l], 1)[0] for l in ll]
  print('filtered', filtered)
  print('filtered', next(zip(*filtered)))
  plt.plot(range(1, len(ll) + 1), next(zip(*filtered)), color='red')
  plt.plot(range(1, len(ll) + 1), next(zip(*ll)), color='blue')
  plt.show()


def main_2(args):
  df = pd.read_csv('/Users/arsen/Workspace/macrobase/bench/workflows/moving_gaussian_nov29_grid/11-29-15_32_52/speed=0.00010.csv')
  N, D = df.shape
  unfiltered_centers = []
  filtered_centers = []
  # for i in range(0, N, stepSize):
  i = 0
  stepSize = 100
  x_hat = np.array([[0, 0], [0, 0]])
  Q = np.identity(D)
  H = np.identity(D)
  R = np.identity(D)
  p = np.array([[0, 0], [0, 0]])
  while i < N:
    print('index =', i)
    A = np.array([[1, stepSize], [0, 1]])
    prior_x_hat = np.matmul(A, x_hat)
    prior_p = _sandwich(A, p) + Q
    print('prior_p', prior_p)
    # print(prior_x_hat, prior_p)
    observe = np.mean(df[i:i + stepSize])
    unfiltered_centers.append(observe)
    kalman_gain = np.matmul(np.matmul(prior_p, np.transpose(H)),
                            inv(_sandwich(H, prior_p) + R))

    print('observed loc:', observe)
    z = np.array([observe,
                 (observe - x_hat[0]) / stepSize])
    print('z', z)
    print('prior_x_hat', prior_x_hat)
    print('delta', z[1] * stepSize)
    print('kalman_gain', kalman_gain)
    x_hat = prior_x_hat + np.matmul(kalman_gain, z - np.matmul(H, prior_x_hat))
    print('x_hat', x_hat)
    print(np.identity(D) - np.matmul(kalman_gain, H))
    p = np.matmul(np.identity(D) - np.matmul(kalman_gain, H),
                  prior_p)
    print('p', p)
    filtered_centers.append(x_hat[0])
    i += stepSize
    print('================')
  xx, yy = zip(*unfiltered_centers)
  plt.scatter(xx, yy, color='blue')
  fx, fy = zip(*unfiltered_centers)
  plt.plot(fx, fy, color='red')
  plt.show()


if __name__ == '__main__':
  args = parse_args()
  main(args)
