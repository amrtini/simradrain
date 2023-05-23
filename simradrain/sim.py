import sys
import numpy as np
from scipy.stats import norm, gamma
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import rotate
from simradrain.fft import spectral_sim as sp_sim
import warnings
import winerror
import win32api
import win32job


g_hjob = None


# helper functions for memory limit
def create_job(job_name='', breakaway='silent'):
    hjob = win32job.CreateJobObject(None, job_name)
    if breakaway:
        info = win32job.QueryInformationJobObject(hjob,
                                                  win32job.JobObjectExtendedLimitInformation)
        if breakaway == 'silent':
            info['BasicLimitInformation']['LimitFlags'] |= (
                win32job.JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK)
        else:
            info['BasicLimitInformation']['LimitFlags'] |= (
                win32job.JOB_OBJECT_LIMIT_BREAKAWAY_OK)
        win32job.SetInformationJobObject(hjob,
                                         win32job.JobObjectExtendedLimitInformation, info)
    return hjob


def assign_job(hjob):
    global g_hjob
    hprocess = win32api.GetCurrentProcess()
    try:
        win32job.AssignProcessToJobObject(hjob, hprocess)
        g_hjob = hjob
    except win32job.error as e:
        if (e.winerror != winerror.ERROR_ACCESS_DENIED or
                sys.getwindowsversion() >= (6, 2) or
                not win32job.IsProcessInJob(hprocess, None)):
            raise
        warnings.warn('The process is already in a job. Nested jobs are not '
                      'supported prior to Windows 8.')


def limit_memory(memory_limit):
    if g_hjob is None:
        return
    info = win32job.QueryInformationJobObject(g_hjob,
                                              win32job.JobObjectExtendedLimitInformation)
    info['ProcessMemoryLimit'] = memory_limit
    info['BasicLimitInformation']['LimitFlags'] |= (
        win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY)
    win32job.SetInformationJobObject(g_hjob,
                                     win32job.JobObjectExtendedLimitInformation, info)


# simulated rainfall object
class SimRain:
    def __init__(self,
                 dur=None,
                 marginals=None,
                 cov=None,
                 domain=(200, 200),
                 adv=(0, 0),
                 xy_anis=(1, 0),
                 st_anis=1,
                 mem_lim=5):
        """
        Initialises simulated rainfall object, requires default event input parameters to initialise object
        :param dur: Event duration, number of time steps - 5min resolution (int)
        :param marginals: Dictionary of event marginal distributions (dict)
        :param cov: Covariance structure of latent Gaussian field, can be nested structures (str)
        :param domain: Spatial domain size - 1km pixel grid (tuple)
        :param adv: Advection velocity, (vx, vy) - mm/h (tuple)
        :param xy_anis: Spatial anisotropy stretch and rotation, (r, theta) (tuple)
        :param st_anis: Spatiotemporal anisotropy factor (float)
        :param mem_lim: Memory limit in GiB (float)
        """
        self.dur = dur  # number of time steps
        self.marginals = marginals  # prop, k, loc, scale
        self.cov = cov
        self.domain = domain  # (x, y)
        self.adv = adv  # (vel_x, vel_y) m/s
        self.xy_anis = xy_anis
        self.st_anis = st_anis
        self.mem_lim = mem_lim  # memory limit in GiB

        self.nu_shifts = np.zeros([2, self.dur], dtype=int)
        for i in range(2):
            # convert to pixels
            shifts = - (np.arange(self.dur) * self.adv[i] / 1000 * 300)
            # convert to integers
            self.nu_shifts[i] = (shifts - np.min(shifts)).astype(int)

        self.proc_arrays = None
        self.rainfall = None

    def simulate_rain(self,
                      del_proc=True,
                      dur=None,
                      marginals=None,
                      cov=None,
                      domain=None,
                      adv=None,
                      xy_anis=None,
                      st_anis=None,
                      mem_lim=None):
        """
        Simulates rainfall event
        :param del_proc: Delete processing steps (bool)
        :param dur: Event duration, number of time steps - 5min resolution (int)
        :param marginals: Dictionary of event marginal distributions (dict)
        :param cov: Covariance structure of latent Gaussian field, can be nested structures (str)
        :param domain: Spatial domain size - 1km pixel grid (tuple)
        :param adv: Advection velocity, (vx, vy) - mm/h (tuple)
        :param xy_anis: Spatial anisotropy stretch and rotation, (r, theta) (tuple)
        :param st_anis: Spatiotemporal anisotropy factor (float)
        :param mem_lim: Memory limit in GiB (float)
        :return: Rainfall field, self.rainfall field
        """
        if dur is not None:
            self.dur = dur
        if marginals is not None:
            self.marginals = marginals
        if cov is not None:
            self.cov = cov
        if domain is not None:
            self.domain = domain
        if adv is not None:
            self.adv = adv
        if xy_anis is not None:
            self.xy_anis = xy_anis
        if st_anis is not None:
            self.st_anis = st_anis
        if mem_lim is not None:
            self.mem_lim = mem_lim
        if not del_proc:
            self.proc_arrays = {}
        self.rainfall = None

        print("Limiting memory to " + str(self.mem_lim) + "GiB.")

        assign_job(create_job())
        memory_limit = self.mem_lim * 1000 * 1024 * 1024  # mem_lim Gi
        limit_memory(memory_limit)

        # identify correct simulation dimensions (t, y, x)
        adv_field_size = (self.dur, self.domain[1], self.domain[0])

        stat_field_size = (self.dur,
                           self.domain[1] + self.nu_shifts[1].max(),
                           self.domain[0] + self.nu_shifts[0].max())

        a_x = max(stat_field_size[1], stat_field_size[2])

        og_field_size = (int(np.ceil(self.dur * self.st_anis)),
                         int(np.ceil(a_x * 2 * np.ceil(self.xy_anis[0]))),
                         int(np.ceil(a_x * 2)))

        rotated_field_size = (int(np.ceil(self.dur * self.st_anis)), a_x, a_x)

        # large gaussian field with correct correlation structure
        print("Simulating large Gaussian field of size " + str(og_field_size) + ".")
        full_gaussian_field = None
        #try:
        srf = sp_sim.SpectralRandomField(og_field_size, self.cov)
        full_gaussian_field = srf.new_simulation()
        #except:
        #    print("SpectralRandomField failed.")

        if full_gaussian_field is not None:
            if not del_proc:
                self.proc_arrays["isotropic"] = full_gaussian_field

            # function to reproject to stretch a dimension of field
            print("Adding in spatial anisotropy effects.")

            def stretch_field(field, l, axis=1):
                if l == 1:
                    output = field
                else:
                    dims = [np.arange(d) for d in field.shape]

                    interpolation_fun = RegularGridInterpolator(tuple(dims), field)

                    new_dims = []
                    for i in range(3):
                        if i == axis:
                            new_dims.append(np.linspace(dims[i].min(), dims[i].max(), int(dims[i].shape[0] * l)))
                        else:
                            new_dims.append(dims[i])

                    tt, yy, xx = np.meshgrid(new_dims[0], new_dims[1], new_dims[2], indexing="ij")
                    input_dims = np.empty([tt.shape[0], tt.shape[1], tt.shape[2], 3])
                    input_dims[..., 0] = tt
                    input_dims[..., 1] = yy
                    input_dims[..., 2] = xx

                    output = interpolation_fun(input_dims)
                return output

            # shrink y dimension by a factor of 1/lambda_xy (self.xy_anis[0])
            stretched_field = stretch_field(
                full_gaussian_field,
                1 / self.xy_anis[0],
                axis=1)

            # rotate spatially by an angle of -theta_xy (self.xy_anis[1])
            rotated_gaussian_field = rotate(
                stretched_field,
                - self.xy_anis[1],
                axes=(1, 2),
                reshape=False)

            # clip field to size
            r_dim = rotated_gaussian_field.shape
            y_start = int(r_dim[1] / 2) - int(rotated_field_size[1] / 2)
            x_start = int(r_dim[2] / 2) - int(rotated_field_size[2] / 2)
            clipped_gaussian_field = rotated_gaussian_field[
                                     :,
                                     y_start: y_start + rotated_field_size[1],
                                     x_start: x_start + rotated_field_size[2]]

            if not del_proc:
                self.proc_arrays["spatially_anisotropic"] = clipped_gaussian_field

            # shrinking t dimension by a factor of 1/kappa_st (self.st_anis)
            print("Adding in spatio-temporal anisotropy effects.")
            stationary_field = stretch_field(
                clipped_gaussian_field,
                1 / self.st_anis,
                axis=0)
            if not del_proc:
                self.proc_arrays["anisotropic"] = stationary_field

            # Transform marignal rainfall distribution
            print("Transforming marginal distributions.")
            gauss_01 = (stationary_field - np.mean(stationary_field)) / np.std(stationary_field)
            unif = norm.cdf(gauss_01.flatten()).reshape(stationary_field.shape)

            zero_vals = unif <= self.marginals["prop_zero"]
            rain_vals = unif > 1 - self.marginals["prop_nz"] * self.marginals["prop_rain"]
            nz_vals = (~zero_vals) & (~rain_vals)

            to_transform = (unif[rain_vals] - np.min(unif[rain_vals])) / (1 - np.min(unif[rain_vals]))
            gamma_vals = gamma.ppf(to_transform, self.marginals["k"], loc=self.marginals["loc"], scale=self.marginals["scale"])

            stationary_rain = np.full(stationary_field.shape, np.nan)
            stationary_rain[zero_vals] = 0.
            stationary_rain[nz_vals] = 0.3
            stationary_rain[rain_vals] = gamma_vals

            if not del_proc:
                self.proc_arrays["stationary_rainfall"] = stationary_rain

            # shift gaussian field to add in advection
            print("Adding in advection effects.")
            rainfall = np.full([self.dur, self.domain[1], self.domain[0]], np.nan)
            adv_gaussian = np.full([self.dur, self.domain[1], self.domain[0]], np.nan)
            nu_shifts = np.zeros([2, self.dur], dtype=int)
            if any(nu_shifts.flatten() != 0):
                for i in range(2):
                    # convert to pixels
                    shifts = - (np.arange(self.dur) * adv[i] / 1000 * 300)
                    # convert to integers
                    nu_shifts[i] = (shifts - np.min(shifts)).astype(int)

            for t in range(self.dur):
                x_min, y_min = nu_shifts[:, t]
                subset_rain = stationary_rain[
                              t,
                              y_min: y_min + self.domain[1],
                              x_min: x_min + self.domain[0]]
                subset_gaus = stationary_field[
                              t,
                              y_min: y_min + self.domain[1],
                              x_min: x_min + self.domain[0]]
                rainfall[t, :, :] = subset_rain
                adv_gaussian[t, :, :] = subset_gaus

            gauss_01 = (adv_gaussian - np.mean(adv_gaussian)) / np.std(adv_gaussian)
            unif = norm.cdf(gauss_01.flatten()).reshape(adv_gaussian.shape)

            zero_vals = unif <= self.marginals["prop_zero"]
            rain_vals = unif > 1 - self.marginals["prop_nz"] * self.marginals["prop_rain"]
            nz_vals = (~zero_vals) & (~rain_vals)

            to_transform = (unif[rain_vals] - np.min(unif[rain_vals])) / (1 - np.min(unif[rain_vals]))
            gamma_vals = gamma.ppf(to_transform, self.marginals["k"], loc=self.marginals["loc"], scale=self.marginals["scale"])

            adv_rainfall = np.full(adv_gaussian.shape, np.nan)
            adv_rainfall[zero_vals] = 0.
            adv_rainfall[nz_vals] = 0.3
            adv_rainfall[rain_vals] = gamma_vals

            self.rainfall = adv_rainfall

            print("Rainfall simulation complete.")

            return self.rainfall


