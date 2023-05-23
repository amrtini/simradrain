################################################################################
# Name:        Spectral Random Field
# Author:      Philipp Guthke, University Stuttgart, Institute of Modeling
#              Hydraulic and Environmental Systems
# Created:     18.07.2013
# Copyright:   (c) Guthke 2013
# e-mail:      Sebastian.Hoerning@iws.uni-stuttgart.de
################################################################################

import numpy as np
import matplotlib.pyplot as plt
from . import covariance_function
import timeit
import time


def main():

    print('\a\a\a\a Started on %s \a\a\a\a' % time.asctime())
    start = timeit.default_timer()  # to get the runtime of the program

#    covariance_model = '1.0 Mat(9)^1.3'
    covariance_model = '1 Lin(43)'
    srf = SpectralRandomField(
                    domain_size=(600, 600),
                    covariance_model=covariance_model,
                    periodic=False
                    )

    n = 100

    for i in range(n):
        srf.new_simulation()

    stop = timeit.default_timer()  # Ending time
    print('\n\a\a\a Done with everything on %s. Total run time was about %0.4f seconds \a\a\a' %
          (time.asctime(), stop-start))


class SpectralRandomField(object):
    """
    This class simulates gaussian random fields with given correlation structure
    using the Fast Fourier Transformation.
    """
    def __init__(self,
                 domain_size=(100, 100),
                 covariance_model='1.0 Exp(2.)',
                 periodic=True,
                 ):

        self.counter = 0
        self.periodic = periodic
        # adjust domain size by cutoff for non-periodic output
        self.cutoff = 0
        if not self.periodic:
            cutoff = covariance_function.find_maximum_range(covariance_model)
            self.cutoff = int(cutoff)

        self.domain_size = np.array(domain_size) + self.cutoff
        self.covariance_model = covariance_model
        self.number_of_dimensions = len(self.domain_size)
        self.number_of_points = np.prod(self.domain_size)

        self.grid = np.mgrid[[slice(0, n, 1) for n in self.domain_size]]

        # ensure periodicity of domain
        for i in range(self.number_of_dimensions):
            self.domain_size = self.domain_size[:, np.newaxis]

        self.grid = np.min((self.grid, np.array(self.domain_size) - self.grid), axis=0)

        # compute distances from origin (--> wave numbers in fourier space)
        self.distances_from_origin = ((self.grid ** 2).sum(axis=0)) ** 0.5

        # covariances (in fourier space!!!)
        self.covariance = covariance_function.covariogram(self.distances_from_origin, self.covariance_model)

        # FFT of covariances
        self.fft_of_covariance = np.abs(np.fft.fftn(self.covariance))

        # eigenvalues of decomposition
        self.eigenvalues = np.sqrt(self.fft_of_covariance / self.number_of_points)

        self.y = self.new_simulation()

    def new_simulation(self):
        self.counter += 1
        # compute random field via inverse fourier transform
        real = np.random.standard_normal(size=self.eigenvalues.shape)
        image = np.random.standard_normal(size=self.eigenvalues.shape)
        epsilon = real + 1j*image
        rand = epsilon * self.eigenvalues

        self.y = np.real(np.fft.ifftn(rand)) * self.number_of_points

        if not self.periodic:
            # read just domain size to correct size (--> no boundary effects...)
            grid_slice = [slice(0, (self.domain_size - self.cutoff)[i].flatten()[0], 1)
                          for i in range(self.number_of_dimensions)]
            self.y = self.y[grid_slice]
            self.y = self.y.reshape(self.domain_size.flatten() - self.cutoff)

        return self.y

    def get_y(self):
        return self.y

    def plt(self):
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        if self.number_of_dimensions == 1:
            plt.plot(self.y)
        if self.number_of_dimensions == 2:
            plt.imshow(self.y, interpolation='nearest')
            plt.colorbar()
        plt.subplot(122)
        plt.hist(self.y.flatten(), 20)
        plt.show()


if __name__ == '__main__':

    main()
