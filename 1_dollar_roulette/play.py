#!/usr/bin/env python3
"""
Report ratio of win scenarios when playing one dollar at a time.
Win = reach stop-money amount

for default values ratio ~= 0.1084
"""
from __future__ import print_function
import argparse
import random
import numpy as np


def parse_args(*argument_array):
  parser = argparse.ArgumentParser()
  parser.add_argument('--start-money', type=int, default=20)
  parser.add_argument('--stop-money', type=int, default=40)
  parser.add_argument('--num-trials', type=int, default=100)
  args = parser.parse_args(*argument_array)
  return args


def play_roulette(init_money, end_money):
  money = init_money
  wining_odds = 18. / 38
  while money not in (0, end_money):
    money += 1 if random.random() < wining_odds else -1
  return money == end_money


def main(args):
  results = [1. * play_roulette(args.start_money, args.stop_money)
             for i in range(args.num_trials)]
  print(np.mean(results))


if __name__ == '__main__':
  args = parse_args()
  main(args)
