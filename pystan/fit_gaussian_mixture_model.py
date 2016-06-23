from helpers import cached_model
import itertools
from scipy.stats import multivariate_normal
import argparse
import cPickle as pickle
import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd


def parse_args(*argument_list):
  parser = argparse.ArgumentParser()
  parser.add_argument('--csv', required=True)
  parser.add_argument('--vars', nargs='+', required=True)
  parser.add_argument('--num-components', default=3, type=int)
  args = parser.parse_args(*argument_list)
  return args


def log_multi_normal_pdf(x, mean, cov):
  return multivariate_normal(mean=mean, cov=cov).pdf(x)

if __name__ == '__main__':
  args = parse_args()
  D = len(args.vars)
  x = pd.read_csv(args.csv)
  x = x[args.vars]
  stan_data_mappings = {
    'K': args.num_components,
    'D': D,
    'N': x.shape[0],
    'x': x,
    'alpha': 1. * np.ones(args.num_components),
    'm0': x.mean(axis=0),
    'beta0': 1,
    # 'Omega0': np.diag(x.std(axis=0)),
    'Omega0': 100. * np.identity(D),
    'dof0': 15,
  }
  model = cached_model('stan/finite_gaussian_mixture.stan')
  fit = model.sampling(data=stan_data_mappings)
  params = fit.extract()
  theta = params['theta']
  locs = params['loc']
  omegas = params['omega']
  M = len(theta)  # Number of samples (4000 by default)
  xx = np.linspace(-10, 110, 100)
  yy = np.zeros(len(xx))
  inverseM = 1. / M
  print stan_data_mappings['Omega0']
  # pyplot.plot(locs[:, 0])
  pyplot.plot(omegas[:1000, 0][:, 0, 0])
  pyplot.plot(locs[:1000, 0])
  pyplot.title(repr((np.median(omegas[:1000, 0][:, 0, 0]), np.mean(locs[:1000, 0]))))
  print ((np.median(omegas[:1000, 0][:, 0, 0]), np.mean(locs[:1000, 0])))
  # mnorm = [multivariate_normal(mean=l, cov=c)
  #          for ll, cc in zip(locs, omegas)
  #          for l, c in zip(ll, cc)]
  # xx = np.linspace(-5, 5, 10)
  # yy = np.zeros(len(xx))
  # pp = [np.mean([n.pdf(z) for n in mnorm]) for z in xx]
  # pyplot.plot(xx, yy)
  pyplot.show()
  # for s in xrange(M):
  #   if s % 100 == 0:
  #     print s
  #   for k in range(args.num_components):
  #     mnorm = multivariate_normal(mean=locs[s][k], cov=omegas[s][k])
  #     for i in range(len(xx)):
  #       yy += inverseM * mnorm.pdf(xx[i])
  # pyplot.plot(xx, yy)
  print theta[0], locs[0], omegas[0]
  print theta.shape, locs.shape, omegas.shape
  with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
  fit.plot()
