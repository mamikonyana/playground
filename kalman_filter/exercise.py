#!/usr/bin/env python3
"""
Write a Kalman Filter
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
    parser.add_argument('--csv', help='File with the signal to be filtered.',
                        default='noisy_1d.csv')
    args = parser.parse_args(*argument_array)
    return args


class KalmanFilter:

    def __init__(self, measurement_sigma, process_sigma, covariance_prior,
                 location_prior):
        # TODO: Initialize
        pass

    def step(self, observation, delta_t=1.):
        H = np.array([[1, 0]])
        F = np.array([[1, delta_t], [0, 1]])
        # TODO: Implement
        next_prediction = np.zeros(observation.shape)  # FIXME
        return next_prediction


def main(args):
    df = pd.read_csv(args.csv)
    unfiltered = [np.array([row['XX']]) for i, row in df.iterrows()]
    # TODO: Filter the signal and plot
    kf = KalmanFilter()  # FIXME
    filtered = [kf.step(x) for x in unfiltered]


if __name__ == '__main__':
  args = parse_args()
  main(args)
