import pystan
from hashlib import md5
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
from helpers.io import extract_from_candles


def cached_stan_model(model_code, model_name=None, **kwargs):
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    if model_name is None:
        cache_fn = 'anon_model_{}.pkl'.format(code_hash)
        contents_cache_fn = 'anon_model_{}.stan'.format(code_hash)
    else:
        cache_fn = '{}_{}.pkl'.format(model_name, code_hash)
        contents_cache_fn = '{}_{}.stan'.format(model_name, code_hash)
    cache_fn = os.path.join('.cache', cache_fn)
    contents_cache_fn = os.path.join('.cache', contents_cache_fn)
    os.makedirs(os.path.dirname(cache_fn), exist_ok=True)
    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except Exception as e:
        print(e)
        sm = pystan.StanModel(model_code=model_code)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
        with open(contents_cache_fn, 'w') as f:
            f.write(contents)
    else:
        print("Using cached StanModel")
    return sm


def _plat_1_and_save(fit, name):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(np.exp(fit.extract()['h'].mean(axis=0)) * 1000, color='magenta',
            label='volatility', alpha=.8)
    ax.plot(np.array(y), color='blue', label='log return', alpha=.3)
    ax.set(xlabel='time', ylabel='log returns', ylim=[-0.02, 0.04])
    ax.legend()
    ax.plot((fp - np.mean(fp)) / np.std(fp) * 0.005, label='ETH Nov1-16',
            color='cyan', zorder=1)
    ax.plot(5 * fit.extract()['m'].mean(axis=0), color='green',
            label='mean log return', alpha=.9)
    ax.grid()
    fig.savefig(name)


if __name__ == '__main__':
    model_filename = 'stochastic_volatility.stan'
    contents = open(model_filename, 'r').read()
    model = cached_stan_model(contents)
    filename = 'tmp/candles_eth_usd_200k_nov1_16.txt.gz'
    [prices, returns] = extract_from_candles(filename, 'close', 'return')
    log_returns = [np.log(1 + float(r)) for r in returns]
    float_returns = [float(r) for r in returns]
    float_prices = [float(p) for p in prices]

    y = log_returns[1500:3500]
    fp = float_prices[1500:3500]

    samples = model.sampling(data={'T': len(y), 'y': y},
                             iter=1000,
                             chains=1)
    for var in ('xi', 'phi'):
        print(var, samples.extract()[var].mean(axis=0))
    code_hash = md5(contents.encode('ascii')).hexdigest()
    _plat_1_and_save(samples, code_hash)
