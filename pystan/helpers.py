import pystan
import pickle
from hashlib import md5


def cached_model(filename):
  """Use just as you would `stan`"""
  with open(filename, 'rb') as infile:
    code_hash = md5(infile.read()).hexdigest()
  cache_fn = 'cached-model-{}.pkl'.format(code_hash)
  try:
    sm = pickle.load(open(cache_fn, 'rb'))
  except:
    sm = pystan.StanModel(file=filename)
    with open(cache_fn, 'wb') as f:
        pickle.dump(sm, f)
  else:
      print("Using cached StanModel")
  return sm
