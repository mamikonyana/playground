#!/usr/bin/env python3
"""
"""
from __future__ import print_function
import argparse
import math
import numpy as np
import random


def parse_args(*argument_array):
  parser = argparse.ArgumentParser()
  parser.add_argument('-n', '--num-points', type=int, default=4)
  parser.add_argument('--trials', type=int, default=10000)
  args = parser.parse_args(*argument_array)
  return args


def angle_less_than_pi(angles):
  angles = sorted(angles)
  for i in range(1, len(angles)):
    if angles[i] - angles[i - 1] > math.pi:
      return True
  return angles[-1] - angles[0] < math.pi


def main(args):
  two_pi = 2 * math.pi
  results = [angle_less_than_pi([random.uniform(0, two_pi)
                                 for _ in range(args.num_points)])
             for trial in range(args.trials)]
  print(np.mean(results))

if __name__ == '__main__':
  args = parse_args()
  main(args)
