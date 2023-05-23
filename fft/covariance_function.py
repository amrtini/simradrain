################################################################################
# Name:        Covariance Functions
# Author:      Philipp Guthke, University Stuttgart,
#              Institute of Modeling Hydraulic and Environmental Systems
# Created:     21.05.2011
# Copyright:   (c) Guthke 2011
# e-mail:      Sebastian.Hoerning@iws.uni-stuttgart.de
################################################################################
import numpy as np
import scipy.special
import pylab as plt


def main():
    h = np.linspace(0, 3, 31)

    a = 1  # Range

    c1 = type_mat(h, model_range=a, v=1.0)
    c2 = type_hol(h, model_range=a)
    c3 = type_exp(h, model_range=a)
    c4 = type_gau(h, model_range=a)
    c5 = type_sph(h, model_range=a)

    plt.figure(figsize=(7, 4))
    plt.plot(h, c3, '--', color='black', label='exponential')
    plt.plot(h, c4, '-.', color='black', label='gaussian')
    plt.plot(h, c5, '-', color='black', label='spherical')
    plt.plot(h, c1, '.-',  color='black', label='matern')
    plt.plot(h, c2, ':',  color='black', label='hole effect')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.xlabel('$h$')
    plt.ylabel('$\\rho$')
    plt.ylim(-0.3, 1)
    plt.tight_layout()
    plt.show()


def covariogram(h, model='1.0 Exp(1.0)'):
    """
    h... distance vector
    model...gstat like string
        *possible models:
            Hol = Hole-effect (Exponential times cosinus)
            Mat = Matern
            Exp = Exponential
            Sph = Spherical
            Gau = Gaussian
            Lin = Linear
            Nug = Nugget
            Pow = Power-law
            Cau = Cauchy
            e.g.: '1.0 Exp(3.7) + 1.9 Mat(2.2)^0.5 + 0.3 Nug(666)'
        *the matern and hole model require an additional parameter:
            'sill Mat(model_range)^parameter'
        *the nugget model requires a model_range also, but it is not taken into account!
            'sill Nug(0)'
        *every other model:
            'sill Typ(model_range)''
        *superposition is possiblewith ' + '
    """
    h = np.atleast_1d(np.array(h).astype(float))

    # check models
    models = model.split('+')
    models = np.array(models)

    # go through models:
    c = np.zeros(h.shape)
    for submodel in models:
        submodel = submodel.strip()
        sill = submodel.split('(')[0].strip()[:-3].strip()
        model_range = submodel.split('(')[1].split(')')[0]
        model_type = submodel.split('(')[0].strip()[-3:]

        sill = float(sill)
        if sill <= 0:
            sill = 0

        model_range = np.abs(np.array(model_range).astype('float'))
        if model_range <= 0:
            model_range = np.array([0.0])

        model_type = np.array(model_type)

        # calculate covariance:
        if model_type == 'Mat':
            param = submodel.split('^')[1].strip()
            param = float(param)
            c0 = c[np.where(h == 0)]
            c += type_mat(h, v=param, model_range=model_range, model_sill=sill)
            c[np.where(h == 0)] = c0 + sill
        elif model_type == 'Hol':
            c += type_hol(h, model_range=model_range, sill=sill)
        elif model_type == 'Exp':
            c += type_exp(h, model_range=model_range, sill=sill)
        elif model_type == 'Sph':
            c += type_sph(h, model_range=model_range, sill=sill)
        elif model_type == 'Gau':
            c += type_gau(h, model_range=model_range, sill=sill)
        elif model_type == 'Lin':
            c += type_lin(h, model_range=model_range, sill=sill)
        elif model_type == 'Nug':
            c[np.where(h == 0)] += sill
        elif model_type == 'Pow':
            c0 = c[np.where(h == 0)]
            c += type_power(h, model_range=model_range, sill=sill)
            c[np.where(h == 0)] = c0 + sill
            print('Not sure if it works yet!')
        elif model_type == 'Cau':
            alpha = submodel.split('^')[1].strip()
            alpha = float(alpha)
            beta = submodel.split('^')[2].strip()
            beta = float(beta)
            c += type_cauchy(h,
                             model_range=model_range,
                             sill=sill,
                             alpha=alpha,
                             beta=beta)

    return c


def type_hol(h, model_range=1.0, sill=1.0):
    h = np.array(h)
    c = np.ones(h.shape) * sill
    ix = np.where(h > 0)
    c[ix] = sill * (np.sin(np.pi * h[ix] / model_range) / (np.pi * h[ix] / model_range))
    return c


def type_exp(h, model_range=1.0, sill=1.0):
    h = np.array(h)
    return sill * (np.exp(-h / model_range))


def type_sph(h, model_range=1.0, sill=1.0):
    h = np.array(h)
    return np.where(h > model_range, [0],
                    sill * (1 - 1.5 * h / model_range + h ** 3 / (2 * model_range ** 3)))


def type_gau(h, model_range=1.0, sill=1.0):
    h = np.array(h)
    return sill * np.exp(-h ** 2 / model_range ** 2)


def type_lin(h, model_range=1.0, sill=1.0):
    h = np.array(h)
    return np.where(h > model_range, [0], sill * (-h / model_range + 1))


def type_mat(h, v=0.5, model_range=1.0, model_sill=1.0):
    """
    Matern Covariance Function Family:
        v = 0.5 --> Exponential Model
        v = inf --> Gaussian Model
    """
    h = np.array(h)

    # for v > 100 shit happens --> use Gaussian model
    if v > 100:
        c = type_gau(h, model_range=1.0, sill=1.0)
    else:
        kv = scipy.special.kv      # modified bessel function of second kind of order v
        tau = scipy.special.gamma  # Gamma function

        fac1 = h / model_range * 2.0 * np.sqrt(v)
        fac2 = (tau(v)*2.0**(v-1.0))
        c = model_sill * 1.0 / fac2 * fac1 ** v * kv(v, fac1)

        # set nan-values at h=0 to sill
        c[np.where(h == 0)] = model_sill

    return c


def type_power(h, model_range=1.0, sill=1.0):
    h = np.array(h)
    return sill - h ** model_range


def type_cauchy(h, model_range=1.0, sill=1.0, alpha=1.0, beta=1.0):
    """
    alpha >0 & <=2 ... shape parameter
    beta >0 ... parameterises long term memory
    """
    h = np.array(h).astype('float')
    return sill * (1 + (h / model_range) ** alpha) ** (-beta / alpha)


def find_maximum_range(model='1.0 Exp(1.0)', rho_thresh=0.01):
    """
    returns range of the model where correlation is rho_thresh
    """
    # check models
    models = model.split('+')
    models = np.array(models)
    # go through models:
    max_range = 0
    for submodel in models:
        submodel = submodel.strip()
        model_range = submodel.split('(')[1].split(')')[0]
        model_range = float(model_range)
        if max_range < model_range:
            max_range = model_range

    # search integral scale...
    integral_scale = 0
    correlation = covariogram(integral_scale, model=model)
    while correlation > rho_thresh:
        integral_scale += max_range/10.0
        correlation = covariogram(integral_scale, model=model)

    integral_scale = max(max_range*3,   integral_scale)
    integral_scale = min(max_range*100, integral_scale)
    return integral_scale


if __name__ == '__main__':
    import timeit
    import time
    
    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    start = timeit.default_timer()  # to get the runtime of the program

    main()
    
    stop = timeit.default_timer()  # Ending time
    print('\n\a\a\a Done with everything on %s. Total run time was about %0.4f seconds \a\a\a' %
          (time.asctime(), stop-start))
