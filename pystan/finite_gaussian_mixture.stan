data {
	int<lower = 2> K; // num clusters
	int<lower = 1> D; // dimensions of the data
	int<lower = 0> N; // number of observations
	vector[D] x[N]; 

	// Dirichlet prior for clusters
	vector[K] alpha;

	// Normal-Wishart Prior
	vector[D] m0;
	real beta0;
	cov_matrix[D] Omega0;
	real dof0;
}

transformed data {
    cov_matrix[D] invOmega0;
    invOmega0 <- inverse(Omega0);
}

parameters {
	simplex[K] theta;  // mixing proportions
	vector[D] loc[K];
	cov_matrix[D] omega[K];
}

model {

  theta ~ dirichlet(alpha);  // prior

  for (k in 1:K) {
    omega[k] ~ inv_wishart(dof0, invOmega0);
    loc[k] ~ multi_normal(m0, omega[k] ./ beta0);
	}

  for (n in 1:N) {
    real gamma[K];
    for (k in 1:K) 
      gamma[k] <- log(theta[k]) + multi_normal_log(x[n], loc[k], omega[k]);
    increment_log_prob(log_sum_exp(gamma));  // likelihood
  }
}
