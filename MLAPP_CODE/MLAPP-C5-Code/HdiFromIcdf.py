from scipy.optimize import fminbound
def hdi_from_icdf(icdf, mass):
    width_fn = lambda lower: icdf(lower+mass)-icdf(lower)
    lower = fminbound(width_fn, 0, 1-mass)
    H = [icdf(lower), icdf(lower + mass)]
    return H
