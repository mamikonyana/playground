#!/usr/bin/env python3
"""
Simulate how many steps is it required to start a chess knight from the corner
of the board and move it uniform randomly around to reach back the same
location.
The magic number is 168 :)
"""
from __future__ import print_function
import argparse
import random
import numpy as np


def parse_args(*argument_array):
  parser = argparse.ArgumentParser()
  parser.add_argument('--init-location', default='a1')
  parser.add_argument('--num-trials', type=int, default=100)
  args = parser.parse_args(*argument_array)
  return args


def parse_location(coordinate):
  assert len(coordinate) == 2
  x, y = coordinate
  x = ord(x) - ord('a')
  assert x in range(0, 8)
  y = int(y) - 1
  assert y in range(0, 8)
  return x, y


def is_feasible_loc(loc):
  x, y = loc
  return x in range(0, 8) and y in range(0, 8)


def suggest_move():
  return random.choice([(2, 1),
                        (1, 2),
                        (-1, 2),
                        (-2, 1),
                        (-2, -1),
                        (-1, -2),
                        (1, -2),
                        (2, -1)])


def move(loc):
  x, y = loc
  while True:
    dx, dy = suggest_move()
    new_loc = x + dx, y + dy
    if is_feasible_loc(new_loc):
      return new_loc


def count_moves(init_loc):
  loc = move(init_loc)
  for step in range(2, 100000):
    loc = move(loc)
    if loc == init_loc:
      return step


def main(args):
  init_loc = parse_location(args.init_location)
  print(np.mean([count_moves(init_loc) for trial in range(args.num_trials)]))


if __name__ == '__main__':
  args = parse_args()
  main(args)
