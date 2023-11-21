import h5py
import psutil
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.ndimage import gaussian_filter
from scipy.stats import gamma, multivariate_normal
from scipy.ndimage import rotate
from scipy.interpolate import RegularGridInterpolator
import gstools as gs
import scipy.optimize as optimize
from simradrain.fft import covariance_function as fft_cov
import matplotlib.pyplot as plt


# useful functions

# estimate variogram field
def varioFFT2D(z):
    """
    Calculates the FFT variogram field
    :param z: Field
    :return: Variogram field
    """
    if not memory_check():
        # find closest multiple of 8
        nn = []
        for i in range(z.ndim):
            nn.append(int(np.ceil(((2 * z.shape[i]) - 1.) / 8.) * 8.))
        nn = np.array(nn)

        # create an indicator matrix with 1s for data and 0s for missing
        Z = np.copy(z)
        Z[np.isnan(z)] = 0
        Zid = np.ones(z.shape)
        Zid[np.isnan(z)] = 0

        # compute the number of pairs
        fx = np.fft.fftn(Z, nn)
        fxid = np.fft.fftn(Zid, nn)
        fx2 = np.fft.fftn(Z * Z, nn)

        nh = np.round(np.real(np.fft.ifftn(np.conj(fxid) * fxid)))

        # variogram
        t1 = np.fft.fftn(Z * Zid, nn)
        t2 = np.fft.fftn(Z * Zid, nn)
        t12 = np.fft.fftn(Z * Z, nn)
        gh = np.real(
            np.fft.ifftn(np.conj(fxid) * t12 + np.conj(t12) * fxid - np.conj(t1) * t2 - t1 * np.conj(t2))) / np.maximum(
            nh, 1) / 2

        # reduce matrices to required size
        t = (nn / 2 + 1).astype(int)
        n = z.shape[0]
        p = z.shape[1]
        nh = np.fft.fftshift(nh)[t[0] - n:t[0] + n - 1, t[1] - p:t[1] + p - 1]
        gh = np.fft.fftshift(gh)[t[0] - n:t[0] + n - 1, t[1] - p:t[1] + p - 1]

        return gh


def stretch_field(field, l, axis=1):
    """
    Stretches field in given axis
    :param field: Field to stretch
    :param l: Stretch factor
    :param axis: Axis to stretch
    :return: Stretched field
    """
    if not memory_check():
        if l == 1:
            output = field
        else:
            dims = [np.arange(d) for d in field.shape]

            interpolation_fun = RegularGridInterpolator(tuple(dims), field)

            new_dims = []
            for i in range(3):
                if i == axis:
                    new_dims.append(np.linspace(np.min(dims[i]), np.max(dims[i]), int(dims[i].shape[0] * l)))
                else:
                    new_dims.append(dims[i])

            tt, yy, xx = np.meshgrid(new_dims[0], new_dims[1], new_dims[2], indexing="ij")
            input_dims = np.empty([tt.shape[0], tt.shape[1], tt.shape[2], 3])
            input_dims[..., 0] = tt
            input_dims[..., 1] = yy
            input_dims[..., 2] = xx

            output = interpolation_fun(input_dims)
        return output


def get_variogram(isotropic, lim=None, dims=(0, 1, 2)):
    '''
    Calcuates the 1D variogram from a 2D/£D variogram field
    :param isotropic: Variogram field
    :param lim: Distance limit to calculate gamma values up to
    :param dims: Dimension of variogram field (i.e. 1D/2D/3D, 1/2/3)
    :return: pd.Series of distances (index) and mean variogram values (gamma)
    '''

    dists = np.empty(isotropic.shape)

    if dims == (0, 1, 2):
        if len(isotropic.shape) == 3:
            for i in range(isotropic.shape[0]):
                for j in range(isotropic.shape[1]):
                    for k in range(isotropic.shape[2]):
                        dists[i, j, k] = np.sqrt(
                            (i - isotropic.shape[0] / 2 + 0.5) ** 2 + (j - isotropic.shape[1] / 2 + 0.5) ** 2 + (
                                    k - isotropic.shape[2] / 2 + 0.5) ** 2)
        elif len(isotropic.shape) == 2:
            for i in range(isotropic.shape[0]):
                for j in range(isotropic.shape[1]):
                    dists[i, j] = np.sqrt(
                        (i - isotropic.shape[0] / 2 + 0.5) ** 2 + (j - isotropic.shape[1] / 2 + 0.5) ** 2)
    if dims == (1, 2):
        for i in range(isotropic.shape[1]):
            for j in range(isotropic.shape[2]):
                dists[:, i, j] = np.sqrt(
                    (i - isotropic.shape[1] / 2 + 0.5) ** 2 + (j - isotropic.shape[2] / 2 + 0.5) ** 2)

    if dims == 0:
        for i in range(isotropic.shape[0]):
            dists[i, :, :] = np.sqrt((i - isotropic.shape[0] / 2 + 0.5) ** 2)
    if dims == 1:
        for i in range(isotropic.shape[1]):
            dists[:, i, :] = np.sqrt((i - isotropic.shape[1] / 2 + 0.5) ** 2)
    if dims == 2:
        for i in range(isotropic.shape[2]):
            dists[:, :, i] = np.sqrt((i - isotropic.shape[2] / 2 + 0.5) ** 2)

    if lim is None:
        lim = int(np.nanmax(dists))
    gamma_vals = np.full(lim, np.nan)
    for h in range(lim):
        cond = (dists >= h) & (dists < h + 1)
        gamma_vals[h] = np.nanmean(isotropic[cond])
    gamma_vals = pd.Series(gamma_vals, index=np.arange(lim) + 0.5)
    return gamma_vals



# get range of variogram
def get_range(gamma_field, n_lim=3, min_delta=-0.000001, smoothed=True, sigma=1):
    """
    Estimates the range of a variogram
    :param gamma_field: Variogram field
    :param n_lim: Number of limits to consider
    :param min_delta: Minimum difference between gamma values to be considered
    :param smoothed: Smoothes variogram (bool)
    :param sigma: Smoothing parameter (see gaussian_filter documentation)
    :return: Estimated range of variogram
    """
    if not memory_check():
        if smoothed:
            v = pd.Series(gaussian_filter(gamma_field, sigma=sigma), index=gamma_field.index)
        else:
            v = pd.Series(gamma_field, index=gamma_field.index)

        delta_v = v.diff()
        neg_delta = np.where(delta_v < min_delta)[0]
        r = np.nan
        for lim in range(n_lim + 1, 0, -1):
            for i in range(len(v) - lim):
                if np.sum(np.diff(neg_delta)[i: i + lim]) == lim:
                    r = v.index[neg_delta[i]]
                    break
        if np.isnan(r):
            if v.iloc[0] == v.iloc[-1]:
                r = 1
            elif v.iloc[-1] > v.iloc[0]:
                r = v.index[-1]
        return r

# check memory
def memory_check():
    """
    Memory check function
    :return: Warning of memory limit exceeds 85%
    """
    mem = psutil.virtual_memory()
    if mem.percent > 85: # change
        warning = True
        print("Memory reached over 80%, stopping code.")
    else:
        warning = False
    return warning


class Event:
    def __init__(
            self,
            file_path
    ):
        self.file_path = file_path

        # read in hdf5 event file
        print("Reading in event data.")
        if self.file_path.endswith(".hdf5"):
            
            with h5py.File(self.file_path, "r") as hf:
                self.arrays = hf["rainfall"][:]
        else:
            self.arrays = np.load(self.file_path)
        
        # allocate attributes
        self.raw_advection = None
        self.advection = None
        self.stationary = None
        self.rainfall_marginals = None
        self.non_zero = None
        self.event_indicator = None
        self.centered_rainfall = None
        self.empirical_variogram = None
        self.spatial_anistopy_est = None
        self.lambda_xy = None
        self.theta_xy = None
        self.marginal_s_vario = None
        self.marginal_t_vario = None
        self.kappa_st = None
        self.spatially_isotropic_vario = None
        self.isotropic_vario = None
        self.marginal_vario = None

    def est_advection(self,
                      thresh=0.001,  # threshold for setting NaNs
                      w=20,  # template size
                      search_w=12  # search window in each direction
                      ):

        print("Estimating advection.")
        ev = np.full(self.arrays.shape, np.nan)
        ev[self.arrays > thresh] = self.arrays[self.arrays > thresh]

        width = ev.shape[1]
        temp_starts = np.arange(w, width - w, w)
        output = []

        for t in range(ev.shape[0] - 1):
            for x_start in temp_starts:
                x_centre = x_start + int(w / 2)
                for y_start in temp_starts:
                    y_centre = y_start + int(w / 2)

                    temp = ev[t, y_start:y_start + w, x_start:x_start + w]  # template from t0
                    corrs = np.full([search_w * 2 + 1, search_w * 2 + 1], np.nan)  # output correlation

                    if np.nansum(temp) > 12:  # at least 15% non-zero pixels

                        for y_i in range(-search_w, search_w + 1):
                            for x_i in range(-search_w, search_w + 1):

                                window = ev[t + 1, y_start + y_i: y_start + w + y_i,
                                         x_start + x_i: x_start + w + x_i]  # search window

                                if np.nansum(window) > 12:  # at least 15% non-zero pixels

                                    # rank and mask data
                                    temp_ranks = np.full(temp.shape, np.nan)
                                    temp_ranks[~np.isnan(temp)] = rankdata(temp[~np.isnan(temp)], "min")
                                    ma_temp = np.ma.masked_where(np.isnan(temp_ranks), temp_ranks)

                                    window_ranks = np.full(window.shape, np.nan)
                                    window_ranks[~np.isnan(window)] = rankdata(window[~np.isnan(window)], "min")
                                    ma_window = np.ma.masked_where(np.isnan(window), window_ranks)
                                    if np.product(ma_temp.shape) == np.product(ma_window.shape):
                                        corr = np.ma.corrcoef(ma_temp.flatten(), ma_window.flatten())[0, 1]
                                        corrs[y_i + search_w, x_i + search_w] = corr

                        corrs[abs(corrs) == 1.] = np.nan
                        rank_corrs = np.full(corrs.shape, np.nan)
                        rank_corrs[~np.isnan(corrs)] = rankdata(corrs[~np.isnan(corrs)], method="ordinal")

                        x_corr = y_corr = max_corr = np.nan
                        if not np.isnan(np.nanmax(rank_corrs)):
                            for rank in range(int(np.nanmax(rank_corrs)), 1, -1):
                                y, x = np.where(rank_corrs == rank)

                                if not any(np.isnan(corrs[max(x[0] - 1, 0): min(x[0] + 2, 25),
                                                    max(y[0] - 1, 0): min(y[0] + 2, 25)]).flatten()):
                                    x_corr = x
                                    y_corr = y
                                    max_corr = corrs[y, x]
                                    break

                        output.append([t, x_centre, y_centre, x_corr - search_w, y_corr - search_w, max_corr])

        # output
        output = pd.DataFrame(output)

        output.columns = ["t", "x_centre", "y_centre", "x_corr", "y_corr", "max_corr"]

        self.raw_advection = output.dropna()  # in km/5min
        for attr in ["x_corr", "y_corr", "max_corr"]:
            self.raw_advection[attr] = self.raw_advection[attr].astype(str).str.strip("[]").str.strip(" ").astype(float)

        vel_x_pixels, vel_y_pixels = self.raw_advection.groupby("t")[["x_corr", "y_corr"]].median().mean()
        vel_x_mps = vel_x_pixels * 1000 / 300  # m/s
        vel_y_mps = vel_y_pixels * 1000 / 300  # m/s

        self.advection = pd.Series([vel_x_pixels, vel_y_pixels, vel_x_mps, vel_y_mps],
                                   index=["vel_x_pixels", "vel_y_pixels", "vel_x_mps", "vel_y_mps"])

    def get_stationary_event(self):
        if self.advection is None:
            self.est_advection()

        print("Removing advection for stationary event.")

        # get stationary event arrays
        vel_x, vel_y = self.advection.loc[["vel_x_pixels", "vel_y_pixels"]]
        x_vels = np.array([vel_x] * self.arrays.shape[0])
        y_vels = np.array([vel_y] * self.arrays.shape[0])

        x_shifts = (np.nancumsum(- x_vels) - np.nancumsum(- x_vels).min()).astype(int)
        y_shifts = (np.nancumsum(- y_vels) - np.nancumsum(- y_vels).min()).astype(int)

        x_max = int(x_shifts.max())
        y_max = int(y_shifts.max())

        # remove advection
        self.stationary = np.full([self.arrays.shape[0],
                                   self.arrays.shape[1] + y_max,
                                   self.arrays.shape[2] + x_max],
                                  np.nan)

        for i in range(self.arrays.shape[0]):
            x_shift = int(x_shifts[i])
            y_shift = int(y_shifts[i])

            self.stationary[i, y_shift: y_shift + 200, x_shift: x_shift + 200] = self.arrays[i]

    def fit_marginal_rainfall(self,
                              zero_thresh=0.1,
                              rain_thresh=0.5,
                              p=0.5):

        if self.stationary is None:
            self.get_stationary_event()

        print("Fitting marginal rainfall distribution.")

        self.rainfall_marginals = {
            "p": p,
            "u": np.percentile(self.stationary[~np.isnan(self.stationary)], p * 100),
            "zero_thresh": zero_thresh,
            "rain_thresh": rain_thresh
        }

        self.non_zero = np.full(self.stationary.shape, np.nan)
        self.non_zero[self.stationary > self.rainfall_marginals["u"]] = self.stationary[
            self.stationary > self.rainfall_marginals["u"]]
        self.non_zero[self.stationary < self.rainfall_marginals["u"]] = 0

        non_missing = self.arrays[~np.isnan(self.arrays)].shape[0]

        self.rainfall_marginals["prop_zero"] = \
        self.arrays[(self.arrays < zero_thresh) & (~np.isnan(self.arrays))].shape[0] / non_missing
        self.rainfall_marginals["prop_nz"] = \
        self.arrays[(self.arrays >= zero_thresh) & (self.arrays < rain_thresh) & (~np.isnan(self.arrays))].shape[
            0] / non_missing
        self.rainfall_marginals["prop_rain"] = \
        self.arrays[(self.arrays >= rain_thresh) & (~np.isnan(self.arrays))].shape[0] / non_missing

        rainfall = self.arrays[(self.arrays >= rain_thresh) & (~np.isnan(self.arrays))]
        k, loc, scale = gamma.fit(rainfall, method="MM")

        self.rainfall_marginals["k"] = k
        self.rainfall_marginals["loc"] = loc
        self.rainfall_marginals["scale"] = scale

    def centre_rainfall(self):

        if self.stationary is None:
            self.get_stationary_event()

        if self.non_zero is None:
            self.fit_marginal_rainfall()

        print("Centering rainfall.")

        # center rainfall
        sum_field_xy = np.nansum(self.non_zero, axis=0)
        smoothed_xy = gaussian_filter(sum_field_xy, sigma=10)

        centre_dims = [int(np.mean(d)) for d in np.where(smoothed_xy == np.nanmax(smoothed_xy))]
        new_dims = np.empty(2, dtype=int)
        new_start = np.empty(2, dtype=int)
        for i in range(2):
            if centre_dims[i] > (self.stationary.shape[i + 1] / 2):
                new_dims[i] = 2 * centre_dims[i]
                new_start[i] = 0
            else:
                new_dims[i] = 2 * (self.stationary.shape[i + 1] - centre_dims[i])
                new_start[i] = new_dims[i] - self.stationary.shape[i + 1]

        self.centered_rainfall = np.full((self.stationary.shape[0], new_dims[0], new_dims[1]), np.nan)
        self.centered_rainfall[:,
        new_start[0]: new_start[0] + self.stationary.shape[1], new_start[1]:
                                                               new_start[1] + self.stationary.shape[
                                                                   2]] = self.stationary

    def get_indicator_variogram(self):

        if self.centered_rainfall is None:
            self.centre_rainfall()

        # get the event indicator
        self.event_indicator = np.full(self.centered_rainfall.shape, np.nan)
        nz_cond = self.centered_rainfall > self.rainfall_marginals["u"]
        nz_rainfall = self.centered_rainfall[nz_cond]

        self.event_indicator[nz_cond] = 1.
        self.event_indicator[self.centered_rainfall <= self.rainfall_marginals["u"]] = 0.

        print("Calculating variogram field.")

        self.empirical_variogram = varioFFT2D(self.event_indicator)

    def get_spatial_anis(self,
                         angles=np.arange(-90, 90, 10),
                         stretches=np.arange(1.5, 10.5, 0.5),
                         lim=75):

        if self.empirical_variogram is None:
            self.get_indicator_variogram()

        print("Estimating the spatial anisotropy.")

        def get_weights(v, lim=lim):
            if not memory_check():
                centre = [int(d / 2) for d in v.shape]
                dist_sq = np.full(v.shape, np.nan)
                for i in range(v.shape[0]):
                    for j in range(v.shape[1]):
                        for k in range(v.shape[2]):
                            dist = np.sqrt((i - centre[0]) ** 2 + (j - centre[1]) ** 2 + (k - centre[2]) ** 2)
                            if dist < lim:
                                dist_sq[i, j, k] = dist
                return 1 / (1 + dist_sq)

        d = int(lim * 1.5)

        v = self.empirical_variogram[:,
            int(self.empirical_variogram.shape[1] / 2) - d: int(self.empirical_variogram.shape[1] / 2) + d + 1,
            int(self.empirical_variogram.shape[2] / 2) - d: int(self.empirical_variogram.shape[2] / 2) + d + 1]

        weights_full = {}
        for stretch in stretches:
            weights_full[str(stretch)] = None

        weights_full["1"] = get_weights(v, lim=lim)
        weights = weights_full["1"]

        optim_mean_sq1 = np.average(
            abs(v[~np.isnan(weights)] - v.transpose(0, 2, 1)[
                ~np.isnan(weights.transpose(0, 2, 1))]),
            weights=weights[~np.isnan(weights)])

        optim_mean_sq2 = np.average(
            abs(v[~np.isnan(weights)] - np.flip(v.transpose(0, 2, 1), 1)[
                ~np.isnan(weights.transpose(0, 2, 1))]),
            weights=weights[~np.isnan(weights)])

        optim_theta1 = 0
        optim_lambda1 = 1
        optim_theta2 = 0
        optim_lambda2 = 1

        print("Estimating spatial anisotropy.")
        for theta in angles:
            print(theta)
            rotated = rotate(v, theta, axes=(1, 2), reshape=False)

            for stretch in stretches:
                if not memory_check():
                    stretched_full = stretch_field(rotated, stretch, axis=1)
                    y_start = int((stretched_full.shape[1] - rotated.shape[1]) / 2)
                    stretched = stretched_full[:, y_start: y_start + rotated.shape[1], :]

                    if weights_full[str(stretch)] is None:
                        weights_full[str(stretch)] = get_weights(stretched, lim=lim)

                    weights = weights_full[str(stretch)]

                    av_mean_sq1 = np.average(
                        abs(stretched[~np.isnan(weights)] - stretched.transpose(0, 2, 1)[
                            ~np.isnan(weights.transpose(0, 2, 1))]),
                        weights=weights[~np.isnan(weights)])

                    av_mean_sq2 = np.average(
                        abs(stretched[~np.isnan(weights)] - np.flip(stretched.transpose(0, 2, 1), 1)[
                            ~np.isnan(weights.transpose(0, 2, 1))]),
                        weights=weights[~np.isnan(weights)])

                    if av_mean_sq1 < optim_mean_sq1:
                        optim_mean_sq1 = av_mean_sq1
                        optim_theta1 = theta
                        optim_lambda1 = stretch

                    if av_mean_sq2 < optim_mean_sq2:
                        optim_mean_sq2 = av_mean_sq2
                        optim_theta2 = theta
                        optim_lambda2 = stretch

        names = ["optim_lambda1", "optim_theta1", "optim_mean_sq1", "optim_lambda2", "optim_theta2", "optim_mean_sq2"]
        values = [optim_lambda1, optim_theta1, optim_mean_sq1, optim_lambda2, optim_theta2, optim_mean_sq2]

        self.spatial_anistopy_est = pd.Series(values, index=names)
        self.lambda_xy = self.spatial_anistopy_est.loc["optim_lambda1"]
        self.theta_xy = self.spatial_anistopy_est.loc["optim_theta2"]

        # rotate and stretch to get spatially isotropic field
        rotated = rotate(self.empirical_variogram, self.theta_xy, axes=(1, 2), reshape=True)
        stretched = stretch_field(rotated, l=self.lambda_xy, axis=1)

        self.spatially_isotropic = stretched[:,
                                   int(stretched.shape[1] / 2) - lim: int(stretched.shape[1] / 2) + lim + 1,
                                   int(stretched.shape[2] / 2) - lim: int(stretched.shape[2] / 2) + lim + 1]

        del rotated, stretched

    def get_spatiotemp_anis(self, plot=False,
                            upper_t=200, lower_t=2, dim_t=1,
                            upper_s=1000, lower_s=15, dim_s=2):

        if self.spatial_anistopy_est is None:
            self.get_spatial_anis()

        # estimate the spatiotemporal anisotropy
        print("Calculating marginal variograms.")

        s_dists = np.empty(self.spatially_isotropic[0].shape)
        for ii in range(s_dists.shape[0]):
            for j in range(s_dists.shape[1]):
                s_dists[ii, j] = np.sqrt((ii - self.spatially_isotropic.shape[1] / 2 + 0.5) ** 2 + (
                            j - self.spatially_isotropic.shape[2] / 2 + 0.5) ** 2)

        t_dists = abs(np.arange(self.spatially_isotropic.shape[0]) - self.spatially_isotropic.shape[0] / 2 + 0.5)

        vario_s0 = self.spatially_isotropic[:, s_dists < 2]
        vario_t0 = self.spatially_isotropic[t_dists < 1]

        lim_s = int(s_dists.max() + 1)
        lim_t = int(t_dists.max() + 1)

        gammas_s0 = np.full(lim_t, np.nan)
        for h in range(lim_t):
            cond = (t_dists >= h) & (t_dists < h + 1)
            gammas_s0[h] = np.nanmean(vario_s0[cond])

        gammas_t0 = np.full(lim_s, np.nan)
        for h in range(lim_s):
            cond = (s_dists >= h) & (s_dists < h + 1)
            gammas_t0[h] = np.nanmean(vario_t0[:, cond])

        self.marginal_t_vario = pd.Series(gammas_s0, index=np.arange(lim_t) + 0.5)
        self.marginal_s_vario = pd.Series(gammas_t0, index=np.arange(lim_s) + 0.5)

        t_fit = None
        s_fit = None

        # get range of variogram
        def get_range(gamma_field, n_lim=3, min_delta=-0.000001, smoothed=True, sigma=1):

            if smoothed:
                v = pd.Series(gaussian_filter(gamma_field, sigma=sigma), index=gamma_field.index)
            else:
                v = pd.Series(gamma_field, index=gamma_field.index)

            delta_v = v.diff()
            neg_delta = np.where(delta_v < min_delta)[0]
            r = np.nan
            for lim in range(n_lim + 1, 0, -1):
                for i in range(len(v) - lim):
                    if np.sum(np.diff(neg_delta)[i: i + lim]) == lim:
                        r = v.index[neg_delta[i]]
                        break
            if np.isnan(r):
                if v.iloc[0] == v.iloc[-1]:
                    r = 1
                elif v.iloc[-1] > v.iloc[0]:
                    r = v.index[-1]
            return r

        def fit_optim_vario(variogram, upper, lower, dim, plot=False,
                            pref=None, lim=0.1, title=None, lim_clip=None):
            models = {
                # "Gaussian": gs.Gaussian,
                "Exponential": gs.Exponential,
                "Spherical": gs.Spherical,
                "Linear": gs.Linear
            }
            if plot:
                fig, ax = plt.subplots()

            v_range = int(get_range(variogram) + 0.5)

            scores = {}
            fit_models = {}
            if plot:
                ax.scatter(variogram.index, variogram, s=5)

            for model in models:

                if model == "Spherical":
                    lower_lim = lower
                    upper_lim = upper
                else:
                    lower_lim = lower / 3
                    upper_lim = upper / 3

                R2 = -1
                MODEL = None

                fit = models[model](dim=dim)
                para, pcov, r2 = fit.fit_variogram(variogram.loc[0: v_range].index, variogram.loc[0: v_range],
                                                   return_r2=True)
                if (fit.len_scale >= lower_lim) & (fit.len_scale <= upper_lim) & (r2 < 1.):
                    R2 = r2
                    MODEL = fit

                if MODEL is None:
                    if lim_clip is not None:
                        clip = lim_clip
                    else:
                        clip = 1
                    for i in range(len(variogram.index) - clip):

                        r = variogram.index[-i]

                        fit_full = models[model](dim=dim)
                        para_full, pcov_full, r2_full = fit_full.fit_variogram(variogram.loc[0: r].index,
                                                                               variogram.loc[0: r], return_r2=True)
                        if (fit_full.len_scale >= lower_lim) & (fit_full.len_scale <= upper_lim) & (r2_full < 1.):
                            if r2_full > r2:
                                R2 = r2_full
                                MODEL = fit_full
                                if MODEL is not None:
                                    break

                if plot:
                    if MODEL is not None:
                        MODEL.plot(ax=ax, alpha=0.3)

                scores[model] = R2
                fit_models[model] = MODEL

            ranking = [(k, v) for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)]
            if pref is not None:

                if abs(scores[pref] - ranking[0][1]) <= lim:

                    optim_model = pref
                else:
                    optim_model = ranking[0][0]

            else:
                optim_model = ranking[0][0]

            model_fit = fit_models[optim_model]

            if plot:
                ax.set_title(title)
                model_fit.plot(ax=ax, linestyle="--")
                plt.show()

            return {"fitted_model": model_fit,
                    "info": [optim_model, ranking[0][1], model_fit.var, model_fit.len_scale, model_fit.nugget]}

        try:
            t_fit = fit_optim_vario(variogram=self.marginal_t_vario.loc[self.marginal_t_vario > 0],
                                    upper=upper_t,
                                    lower=lower_t,
                                    dim=dim_t,
                                    plot=False,
                                    pref="Exponential",
                                    lim=0.05)
        except:
            print("Temporal marginal covariance not fitted.")

        try:
            s_fit = fit_optim_vario(variogram=self.marginal_s_vario.loc[self.marginal_s_vario > 0],
                                    upper=upper_s,
                                    lower=lower_s,
                                    dim=dim_s,
                                    plot=False,
                                    pref="Exponential",
                                    lim=0.05)
        except:
            print("Spatial marginal covariance not fitted.")

        if (s_fit is not None) & (t_fit is not None):

            print("Estimating spatio-temporal anisotropy.")
            s_v = s_fit["fitted_model"]
            t_v = t_fit["fitted_model"]

            if t_fit["info"][0] == "Spherical":
                t_range = int(t_fit["info"][3] * 1.25)
            else:
                t_range = int(t_fit["info"][3] * 1.25 * 3)

            if s_fit["info"][0] == "Spherical":
                s_range = int(s_fit["info"][3] * 1.25)
            else:
                s_range = int(s_fit["info"][3] * 1.25 * 3)

            emp_t = self.marginal_t_vario.loc[0: t_range].dropna()
            emp_s = self.marginal_s_vario.loc[0: s_range].dropna()

            if plot:
                fig, axs = plt.subplots(ncols=2, figsize=(18, 4))

                ax = axs[0]

                s_v.plot(ax=ax, label="s")
                t_v.plot(ax=ax, label="t")

                ax.scatter(emp_s.index, emp_s, s=5)
                ax.scatter(emp_t.index, emp_t, s=5)

            kst_max = 30

            kappas = np.arange(1, kst_max)
            errs = pd.Series(np.nan, index=kappas)

            if plot:
                ax = axs[1]
            for k in kappas:
                if s_range >= t_range:
                    err = abs(pd.Series(s_v.variogram(emp_t.index * k), index=emp_t.index) - emp_t)
                else:
                    err = abs(pd.Series(t_v.variogram(emp_s.index * k), index=emp_s.index) - emp_s)

                errs.loc[k] = err.sum()

            if plot:
                ax.plot(errs)

            if s_range >= t_range:
                kst = kappas[np.argmin(errs)]

                if plot:
                    pd.Series(t_v.variogram(emp_t.index), emp_t.index * kst).plot(linestyle="--", ax=ax, color="red")
                    pd.Series(s_v.variogram(emp_t.index * kst), index=emp_t.index).plot(linestyle="--", ax=ax,
                                                                                        color="black")
            else:
                kst = 1 / kappas[np.argmin(errs)]
                if plot:
                    pd.Series(t_v.variogram(emp_s.index * kst), index=emp_s.index).plot(linestyle="--", ax=ax,
                                                                                        color="black")
                    pd.Series(s_v.variogram(emp_s.index), emp_s.index * kst).plot(linestyle="--", ax=ax, color="red")
                    ax.set_title(kst)

            self.kappa_st = kst

            if plot:
                plt.show()

    def get_isotropic_variogram(self, r=150):

        if self.kappa_st is None:
            self.get_spatiotemp_anis()

        if self.kappa_st is not None:
            # remove spatial anisotropy
            rotated = rotate(self.empirical_variogram, self.theta_xy, axes=(1, 2), reshape=False, cval=np.nan)
            stretched = stretch_field(rotated, l=self.lambda_xy, axis=1)
            c_dim = [int(d / 2) for d in stretched.shape]

            self.spatially_isotropic_vario = stretched[:, c_dim[1] - r: c_dim[1] + r + 1,
                                             c_dim[2] - r: c_dim[2] + r + 1]
            del rotated, stretched

            # convert from indicator to gaussian
            rs = np.arange(0, 1, 0.0001)
            u = self.rainfall_marginals["u"]
            p = self.rainfall_marginals["p"]
            rhos = np.array([multivariate_normal([0, 0], [[1, r], [r, 1]]).cdf([u, u]) for r in rs])

            semi_vario = 0.5 * self.spatially_isotropic_vario
            covar = (p - semi_vario).flatten()
            rescaled_covar = np.array([rs[np.argmin(abs(x - rhos))] for x in covar])
            gaussian_vario = 1 - rescaled_covar.reshape(self.spatially_isotropic_vario.shape)

            # remove spatiotemporal anisotropy
            self.isotropic_vario = stretch_field(gaussian_vario, l=self.kappa_st, axis=0)

            self.marginal_vario = get_variogram(self.isotropic_vario)

    def fit_vario_model(self, r=76, models=["Exp", "Sph", "Gau", "Lin"], nested=True):

        if self.marginal_vario is None:
            self.get_isotropic_variogram()

        model_combs = []

        for model1 in models:
            for model2 in models:
                model_combs.append([model1, model2])
                model_combs.append([model2, model1])

        try:
            v = self.marginal_vario.loc[0:r]

            fun_values = {}
            fitted_eqs = {}
            fitted_gammas = pd.DataFrame(index=v.index)

            for model in models:
                def model_mse(params):  # requires 2 inputs, sill and range
                    sill, range1 = params
                    model_eq = str(round(sill, 3)) + " " + model + "(" + str(round(range1, 1)) + ")"
                    fitted_gamma = sill - fft_cov.covariogram(v.index, model=model_eq)
                    return np.average((v - fitted_gamma) ** 2, weights=1 / (1 + v.index) ** 0.5) ** 0.5

                bounds_model = ((0.001, 1), (1, 1000))
                result_model = optimize.differential_evolution(model_mse, bounds_model)

                sill, range1 = result_model.x
                model_eq = str(round(sill, 3)) + " " + model + "(" + str(round(range1, 1)) + ")"
                fun_values[model_eq] = result_model.fun
                fitted_gammas[model_eq] = sill - fft_cov.covariogram(v.index, model=model_eq)

                def model_nug_mse(params):  # reuires 3 inputs, sill, range and nugget ratio

                    sill, range1, nugget_ratio = params
                    nugget = nugget_ratio * sill
                    sill1 = (1 - nugget_ratio) * sill
                    model_eq = str(round(nugget, 3)) + " Nug(1) + " + str(round(sill1, 3)) + " " + model + "(" + str(
                        round(range1, 1)) + ")"
                    fitted_gamma = sill - fft_cov.covariogram(v.index, model=model_eq)
                    return np.average((v - fitted_gamma) ** 2, weights=1 / (1 + v.index) ** 0.5) ** 0.5

                bounds_model_nug = ((0.001, 1), (1, 1000), (0.001, 0.999))
                result_model_nug = optimize.differential_evolution(model_nug_mse, bounds_model_nug)
                sill, range1, nugget_ratio = result_model_nug.x
                nugget = nugget_ratio * sill
                sill1 = (1 - nugget_ratio) * sill

                model_eq = str(round(nugget, 3)) + " Nug(1) + " + str(round(sill1, 3)) + " " + model + "(" + str(
                    round(range1, 1)) + ")"
                fitted_gammas[model_eq] = sill - fft_cov.covariogram(v.index, model=model_eq)
                fun_values[model_eq] = result_model_nug.fun

            if nested:
                for model_comb in model_combs:
                    def model2_mse(params):  # requires 4 inputs, sill, sill_ratio, range1, range2
                        sill, sill_ratio, range1, range2 = params
                        sill1 = sill * (1 - sill_ratio)
                        sill2 = sill * sill_ratio
                        model_eq = str(round(sill1, 3)) + " " + model_comb[0] + "(" + str(
                            round(range1, 1)) + ") + " + str(round(sill2, 3)) + " " + model_comb[1] + "(" + str(
                            round(range2, 1)) + ")"
                        fitted_gamma = sill - fft_cov.covariogram(v.index, model=model_eq)
                        return np.average((v - fitted_gamma) ** 2, weights=1 / (1 + v.index) ** 0.5) ** 0.5

                    bounds_model2 = (
                    (0.001, 1), (0.001, 0.999), (10, 1000), (1, 10))  # (condition range2 to be small scale variability)
                    result_model2 = optimize.differential_evolution(model2_mse, bounds_model2)

                    sill, sill_ratio, range1, range2 = result_model2.x
                    sill1 = sill * (1 - sill_ratio)
                    sill2 = sill * sill_ratio

                    model_eq = str(round(sill1, 3)) + " " + model_comb[0] + "(" + str(round(range1, 1)) + ") + " + str(
                        round(sill2, 3)) + " " + model_comb[1] + "(" + str(round(range2, 1)) + ")"
                    fitted_gammas[model_eq] = sill - fft_cov.covariogram(v.index, model=model_eq)
                    fun_values[model_eq] = result_model2.fun

                    def model2_nug_mse(params):  # requires 5 inputs, sill, sill_ratio, range1, range2, nugget_ratio

                        sill, sill_ratio, range1, range2, nugget_ratio = params
                        nugget = sill * nugget_ratio
                        sill1 = sill * (1 - sill_ratio) * (1 - nugget_ratio)
                        sill2 = sill * sill_ratio * (1 - nugget_ratio)
                        model_eq = str(round(nugget, 3)) + " Nug(1) + " + str(round(sill1, 3)) + " " + model_comb[
                            0] + "(" + str(round(range1, 1)) + ") + " + str(round(sill2, 3)) + " " + model_comb[
                                       1] + "(" + str(round(range2, 1)) + ")"
                        fitted_gamma = sill - fft_cov.covariogram(v.index, model=model_eq)
                        return np.average((v - fitted_gamma) ** 2, weights=1 / (1 + v.index) ** 0.5) ** 0.5

                    bounds_model2_nug = ((0.001, 1), (0.001, 0.999), (10, 1000), (1, 10),
                                         (0.001, 0.999))  # (condition range2 to be small scale variability)
                    result_model2_nug = optimize.differential_evolution(model2_nug_mse, bounds_model2_nug)
                    sill, sill_ratio, range1, range2, nugget_ratio = result_model2_nug.x
                    nugget = sill * nugget_ratio
                    sill1 = sill * (1 - sill_ratio) * (1 - nugget_ratio)
                    sill2 = sill * sill_ratio * (1 - nugget_ratio)
                    model_eq = str(round(nugget, 3)) + " Nug(1) + " + str(round(sill1, 3)) + " " + model_comb[
                        0] + "(" + str(round(range1, 1)) + ") + " + str(round(sill2, 3)) + " " + model_comb[
                                   1] + "(" + str(round(range2, 1)) + ")"
                    fitted_gammas[model_eq] = sill - fft_cov.covariogram(v.index, model=model_eq)
                    fun_values[model_eq] = result_model2_nug.fun

            self.fitted_vario_model_mse = pd.Series(fun_values)
            self.fitted_model = self.fitted_vario_model_mse.argmin()
        except:
            print("Cannot fit variogram model.")

