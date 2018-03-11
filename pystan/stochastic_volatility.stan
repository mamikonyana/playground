data {
  int<lower=0> T;   // # time points (equally spaced)
  vector[T] y;      // mean corrected return at time t
}

parameters {
  real mu;                     // mean log volatility
  real<lower=-1,upper=1> phi;  // persistence of volatility
  real<lower=0> sigma;         // white noise shock scale
  vector[T] h_std;             // std log volatility time t
  real<lower=0.7, upper=1> xi;   // persistence of mean log return
  vector[T] e;                 // mean log return error
}

transformed parameters {
  vector[T] h;                 // log volatility at time t
  vector[T] m;                 // mean log return at time t
  h = h_std * sigma;
  h[1] = h[1] / sqrt(1 - phi * phi);
  m = 0.0001 * e;
  h = h + mu;
  for (t in 2:T)
    h[t] = h[t] + phi * (h[t-1] - mu);
  for (t in 2:T)
    m[t] = m[t] + xi * m[t-1];
}

model {
  sigma ~ cauchy(0, 0.1);
  mu ~ cauchy(0, 10);  
  h_std ~ normal(0, 0.01);
  e ~ normal(0, 1);
  y ~ normal(m, exp(h / 2));
  xi ~ cauchy(0.9, 0.1);
  phi ~ normal(0.99, 0.01);
}
