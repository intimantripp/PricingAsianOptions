import numpy as np
import scipy


def price_asian_option_mc(S0, K, T, r, sigma, p=1.0, average_type ='arithmetic', num_paths = 10000, num_steps = 252,
                          seed=None):
    dt = T/num_steps
    S = np.empty((num_paths, num_steps + 1))
    S[:, 0] = S0

    # Simulate paths using the geometric Brownian motion model
    Z = np.random.randn(num_paths, num_steps)
    for t in range(1, num_steps + 1):
        S[:, t] = S[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1])

    if p > 0:
        avg_steps = int(np.ceil(p * num_steps))
        avg_steps = max(avg_steps, 1)
    else:
        avg_steps = 0
    
    if p == 0:
        avg_price = S[:, -1]
    else:
        if average_type.lower() == 'arithmetic':
            avg_price = np.mean(S[:, :avg_steps + 1], axis=1)
        

