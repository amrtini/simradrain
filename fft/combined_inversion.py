import numpy as np
import os
import datetime
from ..fft import spectral_sim
import scipy.stats as sct
from scipy.special import ndtri


def five_by_five(z_field, cx, cy):
    av_val = np.average(z_field[cx-2:cx+3, cy-2:cy+3])
    return av_val


def table_np_gauss(a_ll, h):
    n_disc = 4000
    n_dat = a_ll.shape[0]

    a1 = np.min(a_ll) - 3. * h
    a2 = np.max(a_ll) + 3. * h
    delt = (a2 - a1) / n_disc
    b = np.empty(n_disc)
    ball = np.empty(n_dat)
    ball.fill(0.0)
    b.fill(0.0)
    help_array = np.arange(n_disc)
    help_array = help_array * delt + a1
    for j in range(n_dat):
        hp = help_array - a_ll[j]
        hp = 1.0/np.exp(0.5*(hp/h)**2)
        b = b + hp
    cc = 6.28318530718
    b = b/(np.sqrt(cc)*h*n_dat)
    aa = 0.0
    bb = np.empty(2000)
    for i in range(n_disc):
        aa = aa + delt * b[i]
        ch = np.rint(aa*2000.0)
        bb[int(ch):] = np.exp(help_array[i])
    # Find probabilities for observed values
    for j in range(n_dat):
        for k in range(2000):
            if bb[k] > np.exp(a_ll[j]) and ball[j] < 0.0001:
                ball[j] = k/2001
    return ball, bb


def reg_np_gauss(a_ll, b_ll, h):
    n_disc = 1000
    n_dat = a_ll.shape[0]
    print(n_dat)

    a1 = np.min(a_ll) - 2. * h
    a2 = np.max(a_ll) + 2. * h
    delt = (a2 - a1) / n_disc

    b = np.empty(n_disc)
    b.fill(0.0)
    c = np.empty(n_disc)
    c.fill(0.0)
    help_array = np.arange(n_disc)
    help_array = help_array * delt + a1
    for j in range(n_dat):
        hp = help_array - a_ll[j]
        hp = 1.0/np.exp(0.5*(hp/h)**2)
        hq = hp * b_ll[j]
        b = b + hp
        c = c + hq
    b = c / b
    return n_disc, help_array, b


def make_grid_tyne():
    # Wye grid
    ngr = 0

    xy_gr = np.empty([1, 2], dtype=float)
    xy_n = np.empty([1, 2], dtype=float)
    xll_corner = 356000.0
    yll_corner = 535000.0
    for j in range(15):
        xy_n[0, 1] = (14 - j) * 5000.0 + yll_corner
        for i in range(10):
            xy_n[0, 0] = i * 5000.0 + xll_corner
            if ngr > 0:
                xy_gr = np.append(xy_gr, xy_n, 0)
            else:
                xy_gr = np.copy(xy_n)
            ngr = ngr+1
    return xy_gr, ngr


def read_gauge_locations():
    """Returns array of gauge locations"""
    path = os.path.join(os.path.dirname(__file__), 'data', 'Tyne_stats.dat')
    if os.path.isfile("".join(path)):
        f = open("".join(path), 'r')
        lines = f.readlines()
        j = 0
        xy = np.empty([1, 2], dtype=float)
        xyn = np.empty([1, 2], dtype=float)
        for line in lines:
            if j == 0:
                gdata = line.split()
                xy[j, 0] = float(gdata[0])
                xy[j, 1] = float(gdata[1])
            else:
                gdata = line.split()
                xyn[0, 0] = float(gdata[0])
                xyn[0, 1] = float(gdata[1])
                xy = np.append(xy, xyn, 0)
            j = j+1
    else:
        raise Exception('Tyne_stats.dat does not exists')
    return xy


def read_varios():
    path = os.path.join(os.path.dirname(__file__), 'data', 'rcov_Tyne.txt')
    print("".join(path))
    v_pars = np.loadtxt(path, delimiter=',')
    return v_pars


def read_events():
    path = os.path.join(os.path.dirname(__file__), 'data', 'D_Events_Tyne.dat')
    events = np.loadtxt(path, delimiter=' ')
    start = datetime.datetime.strptime("01-01-1999", "%d-%m-%Y")
    for i in range(events.shape[0]):
        et = "%i" % events[i, 3] + "-" + "%i" % events[i, 2] + "-" + "%i" % events[i, 1]
        test = datetime.datetime.strptime(et, "%d-%m-%Y")
        events[i, 0] = (test - start).days
    print(events.shape)
    return events


def read_single_events():
    path = os.path.join(os.path.dirname(__file__), 'data', 'manuEvents_Tyne.dat')
    print("".join(path))
    events = np.loadtxt(path, delimiter=' ')
    start = datetime.datetime.strptime("01-01-1999", "%d-%m-%Y")
    for i in range(events.shape[0]):
        et = "%i" % events[i, 3] + "-" + "%i" % events[i, 2] + "-" + "%i" % events[i, 1]
        test = datetime.datetime.strptime(et, "%d-%m-%Y")
        events[i, 0] = (test - start).days
    return events


def read_b_events():
    path = os.path.join(os.path.dirname(__file__), 'data', 'b4D_Events_Tyne.dat')
    print("".join(path))
    events = np.loadtxt(path, delimiter=' ')
    start = datetime.datetime.strptime("01-01-1999", "%d-%m-%Y")
    for i in range(events.shape[0]):
        et = "%i" % events[i, 4] + "-" + "%i" % events[i, 3] + "-" + "%i" % events[i, 2]
        test = datetime.datetime.strptime(et, "%d-%m-%Y")
        events[i, 1] = (test - start).days
    return events


def read_b_s_events():
    path = os.path.join(os.path.dirname(__file__), 'data', 'S3D5_Events_Tyne.dat')
    print("".join(path))
    events = np.loadtxt(path, delimiter=' ')
    start = datetime.datetime.strptime("01-01-1999", "%d-%m-%Y")
    for i in range(events.shape[0]):
        et = "%i" % events[i, 4] + "-" + "%i" % events[i, 3] + "-" + "%i" % events[i, 2]
        test = datetime.datetime.strptime(et, "%d-%m-%Y")
        events[i, 1] = (test - start).days
    return events


def read_rain():
    """Returns a numpy array read from precipitation data file"""
    path = os.path.join(os.path.dirname(__file__), 'data', 'Allprec.dat')
    print("".join(path))
    rain = np.loadtxt(path, delimiter=' ')
    rain[rain < 0] = -99999
    print(rain.shape)

    return rain


def make_mat(sim_field, i_loc, n_fields):
    # create matrix corresponding to observation locations
    # coefficients are the simulated values at these locations
    res_mat = np.empty([i_loc.shape[0]], dtype=float)
    for i in range(n_fields):
        o_sim = sim_field[i][i_loc[:, 0], i_loc[:, 1]]
        if i == 0:
            res_mat = o_sim
        else:
            res_mat = np.vstack([res_mat, o_sim])
    return res_mat


def daily_rain(rain, lb):
    lin_num = 0
    d_rain_avg = []
    d_rain_min = []
    for lk1 in range(0, lb, 24):
        if int(rain[lk1, 0]) > 1998:
            lin_num = lin_num + 1
            r_act = rain[lk1:lk1 + 24, 4:]
            r_sum = np.sum(r_act, 0)
            r_pos = r_sum[r_sum >= 0]
            rda = np.average(r_pos)
            ram = np.min(r_pos)
            d_rain_avg = np.append(d_rain_avg, rda)
            d_rain_min = np.append(d_rain_avg, ram)
    return d_rain_avg, d_rain_min


def r_first():

    gauge_locations = read_gauge_locations()
    number_of_gauges = gauge_locations.shape[0]

    rain_values = read_rain()
    record_length = rain_values.shape[0]
    #  Tyne
    xll_corner = 356000.0
    yll_corner = 535000.0
    # Wye
    #    xll_corner = 278000.0
    #    yll_corner = 206000.0
    events = read_events()

    dlt = 50000.0

    gauge_locations = gauge_locations - [xll_corner - dlt, yll_corner - dlt]
    ixy = np.rint(gauge_locations / 1000.0).astype(int)
    # Reference grid indices

    v_pars = read_varios()

    smooth_path = os.path.join(os.path.dirname(__file__), 'outputs', 'S_Norms.csv')

    smooth_path_file = open(smooth_path, 'w')

    for i_sim in range(0, 1, 1):
        np.random.seed(seed=i_sim)
        event_index = 0
        year = events[event_index, 1].astype(int)
        month = events[event_index, 2].astype(int)
        day = events[event_index, 3].astype(int)
        print(year, month, day)

        # Do for all selected events
        line_number = 0
        smooth_field = None
        for event_rain_index in range(0, record_length, 24):
            if int(rain_values[event_rain_index, 0]) > 1998:
                line_number = line_number + 1
                if year == rain_values[event_rain_index, 0] \
                        and month == rain_values[event_rain_index, 1] \
                        and day == rain_values[event_rain_index, 2]:
                    
                    gauge_values = np.empty([number_of_gauges])
                    event_rain_values = rain_values[event_rain_index:event_rain_index + 24, 4:]
                    event_daily_totals = np.sum(event_rain_values, 0)
                    positive_event_daily_totals = event_daily_totals[event_daily_totals >= 0]
                    number_of_daily_totals_above_zero = positive_event_daily_totals.shape[0]

                    # Find non-parametric density of log precipitation
                    # Using empirical formula for kernel width
                    # Check if there are any days with no rainfall, if there are, just continue
                    if np.min(positive_event_daily_totals) > 0:
                        logs = np.log(positive_event_daily_totals)
                        standard_deviation = np.std(logs)
                        hw = 0.9 * standard_deviation / number_of_daily_totals_above_zero ** 0.2
                        p_dens, dens = table_np_gauss(logs, hw)
                    else:
                        continue
                    sorted_index_of_event_daily_totals = event_daily_totals.argsort()
                    sorted_index_of_sorted_indices = sorted_index_of_event_daily_totals.argsort()
                    p_rank = sorted_index_of_sorted_indices - number_of_gauges + number_of_daily_totals_above_zero + 1
                    p_rank = p_rank / (number_of_daily_totals_above_zero + 1)

                    k = 0
                    gxy = []
                    for j in range(number_of_gauges):
                        if p_rank[j] > 0:
                            gauge_values[k] = ndtri(p_dens[k])
                            gxy = np.append(gxy, ixy[j])
                            k = k + 1
                    gxy = np.reshape(gxy, [number_of_daily_totals_above_zero, 2])
                    gxy = gxy.astype(int)

                    # Find monthly rank correlation function
                    ccm = '1.0 Sph(%0.1f)' % (v_pars[month - 1, 1] / 1000.)
                    print(ccm)

                    sp_sim = spectral_sim.SpectralRandomField(domain_size=(250, 250), covariance_model=ccm)
                    nf = number_of_daily_totals_above_zero * 30

                    sim = []
                    n_prev = number_of_daily_totals_above_zero * 3
                    norm_sm = 99.9
                    for i in range(nf):
                        sim.append(sp_sim.new_simulation())

                    smooth_field = None
                    while (norm_sm > 0.1) and (n_prev < nf):
                        n_prev = n_prev + 10
                        aa = make_mat(sim, gxy, n_prev)
                        u, s, v = np.linalg.svd(aa)
                        sv = np.zeros([u.shape[0], v.shape[0]], float)
                        sr = np.zeros([v.shape[0], u.shape[0]], float)
                        for i in range(s.shape[0]):
                            sv[i, i] = s[i]
                            if s[i] > 0.01:
                                sr[i, i] = 1. / s[i]
                        hh = np.dot(v.T, np.dot(sr, u.T))

                        #  zz is the target
                        if number_of_gauges == number_of_daily_totals_above_zero:
                            zz = np.copy(gauge_values)
                        else:
                            zz = gauge_values[:-number_of_gauges + number_of_daily_totals_above_zero]
                        c1 = np.dot(hh.T, zz)
                        c1 = c1[:, None, None]
                        norm_sm = np.sum(c1 ** 2)
                        sma = sim[:n_prev] * c1
                        # Smooth field
                        smooth_field = np.sum(sma, axis=0)

                    # Save low norm solution

                    smooth_path = os.path.join(os.path.dirname(__file__), 'outputs' 'Smooth_')
                    smooth_path = smooth_path + '_%i.csv' % line_number
                    assert smooth_field is not None, 'sm_field not defined'
                    np.savetxt(smooth_path, smooth_field, fmt='%0.4f', delimiter=',', newline='\n')
                    # Generate kk homogeneous fields

                    kk = 2
                    sim_hom = []
                    for ii in range(kk):
                        nf = 50
                        sim = []
                        for i in range(nf):
                            sim.append(sp_sim.new_simulation())
                        aa = make_mat(sim, gxy, nf - 5)
                        u, s, v = np.linalg.svd(aa)
                        sv = np.zeros([u.shape[0], v.shape[0]], float)
                        sr = np.zeros([v.shape[0], u.shape[0]], float)
                        for i in range(s.shape[0]):
                            sv[i, i] = s[i]
                            if s[i] > 0.01:
                                sr[i, i] = 1. / s[i]
                        hh = np.dot(v.T, np.dot(sr, u.T))
                        #  zz is the target
                        zz = sim[nf - 1][gxy[:, 0], gxy[:, 1]]
                        c1 = np.dot(hh.T, zz)
                        norm = np.sum(c1 ** 2) + 1.0
                        c1 = c1[:, None, None]
                        sua = sim[:nf - 5] * c1
                        smb = np.sum(sua, axis=0)

                        smb = (smb - sim[nf - 1]) / norm ** 0.5
                        noise_path = os.path.join(os.path.dirname(__file__), 'outputs', 'Noise_%i' % ii)
                        noise_path = noise_path + '_%i.csv' % line_number
                        np.savetxt(noise_path, smb, fmt='%0.4f', delimiter=',', newline='\n')

                        sim_hom.append(smb)
                        nh = "%i" % line_number + "," + "%0.5f" % norm_sm + "\n"
                        smooth_path_file.write(nh)
                    event_index = event_index + 1
                    if event_index <= events.shape[0]:
                        year = events[event_index, 1].astype(int)
                        month = events[event_index, 2].astype(int)
                        day = events[event_index, 3].astype(int)
                    else:
                        year = 9999
                    print(day, month, year, event_index, events.shape[0])


def regular_step():
    gauge_locations = read_gauge_locations()
    number_of_gauges = gauge_locations.shape[0]

    lb, rain = read_rain()
    #  Tyne
    xllcorner = 356000.0
    yllcorner = 535000.0
    # Wye
    #    xllcorner = 278000.0
    #    yllcorner = 206000.0

    events = read_events()
    b_events = read_b_events()

    dlt = 50000.0

    gauge_locations = gauge_locations - [xllcorner - dlt, yllcorner - dlt]
    ixy = np.rint(gauge_locations / 1000.0).astype(int)

    v_pars = read_varios()

    for i_sim in range(0, 1, 1):
        ie = 0
        ev = events[ie, 1].astype(int)
        ho = events[ie, 2].astype(int)
        nap = events[ie, 3].astype(int)
        print(ev, ho, nap)

        # Do for all selected events
        lin_num = 0
        j_paros = True
        for lk1 in range(0, lb, 24):
            if int(rain[lk1, 0]) > 1998:
                lin_num = lin_num + 1
                if ev == rain[lk1, 0] and ho == rain[lk1, 1] and nap == rain[lk1, 2]:
                    ii = 8
                    if j_paros:
                        j1 = b_events[ie//2, 0]//10
                        j_paros = False
                    else:
                        j1 = b_events[ie//2, 0] % 10
                        j_paros = True

                    g_val = np.empty([number_of_gauges])
                    r_act = rain[lk1:lk1 + 24, 4:]
                    r_sum = np.sum(r_act, 0)
                    r_pos = r_sum[r_sum >= 0]
                    n_pos = r_pos.shape[0]
                    rda = np.average(r_pos)
                    print(rda, r_pos)

                    # Find non-parametric density of log precipitation
                    # Using empirical formula for kernel width
                    if np.min(r_pos) > 0:
                        d_pos = np.log(r_pos)
                        dw = np.std(d_pos)
                        hw = 0.9 * dw / n_pos ** 0.2
                        p_dens, dens = table_np_gauss(d_pos, hw)

                    r_sort = r_sum.argsort()
                    r_rank = r_sort.argsort()
                    prank = r_rank - number_of_gauges + n_pos + 1
                    prank = prank / (n_pos + 1)

                    k = 0
                    gxy = []
                    for j in range(number_of_gauges):
                        if prank[j] > 0:
                            g_val[k] = ndtri(p_dens[k])
                            gxy = np.append(gxy, ixy[j])
                            k = k + 1
                    gxy = np.reshape(gxy, [n_pos, 2])
                    gxy = gxy.astype(int)

                    ccm = '1.0 Sph(%0.1f)' % (v_pars[ho - 1, 1] / 1000.)
                    print(ccm)

                    sp_sim = spectral_sim.SpectralRandomField(domain_size=(250, 250), covariance_model=ccm)

                    # Load homogeneous fields
                    path = os.path.join(os.path.dirname(__file__), 'outputs', 'Noise_%i' % ii)
                    path = path + '_%i.csv' % lin_num
                    smb0 = np.loadtxt(path, delimiter=',')

                    ii = ii + 1
                    path = os.path.join(os.path.dirname(__file__), 'outputs', 'Noise_%i' % ii)
                    path = path + '_%i.csv' % lin_num
                    smb1 = np.loadtxt(path, delimiter=',')

                    fi = 2.0 * 3.14159265359 * j1 / 10.0
                    smc = np.cos(fi) * smb0 + np.sin(fi) * smb1
                    ii = ii + 1
                    path = os.path.join(os.path.dirname(__file__), 'outputs', 'Noise_%i' % ii)
                    path = path + '_%i.csv' % lin_num
                    np.savetxt(path, smc, fmt='%0.4f', delimiter=',', newline='\n')

                    nf = 50
                    sim = []
                    for i in range(nf):
                        sim.append(sp_sim.new_simulation())
                    aa = make_mat(sim, gxy, nf - 5)
                    u, s, v = np.linalg.svd(aa)
                    sv = np.zeros([u.shape[0], v.shape[0]], float)
                    sr = np.zeros([v.shape[0], u.shape[0]], float)
                    for i in range(s.shape[0]):
                        sv[i, i] = s[i]
                        if s[i] > 0.01:
                            sr[i, i] = 1. / s[i]
                    hh = np.dot(v.T, np.dot(sr, u.T))
                    #  zz is the target
                    zz = sim[nf - 1][gxy[:, 0], gxy[:, 1]]
                    c1 = np.dot(hh.T, zz)
                    norm = np.sum(c1 ** 2) + 1.0
                    c1 = c1[:, None, None]
                    sua = sim[:nf - 5] * c1
                    smb = np.sum(sua, axis=0)

                    smb = (smb - sim[nf - 1]) / norm ** 0.5
                    ii = ii + 1
                    path = os.path.join(os.path.dirname(__file__), 'outputs', 'Noise_%i' % ii)
                    path = path + '_%i.csv' % lin_num
                    np.savetxt(path, smb, fmt='%0.4f', delimiter=',', newline='\n')

                    ie = ie + 1
                    if ie <= events.shape[0]:
                        ev = events[ie, 1].astype(int)
                        ho = events[ie, 2].astype(int)
                        nap = events[ie, 3].astype(int)
                    else:
                        ev = 9999
                    print(nap, ho, ev, ie, events.shape[0])


def first_mixed_step():

    gauge_locations = read_gauge_locations()
    gauge_locations = gauge_locations.shape[0]

    lb, rain = read_rain()
    #  Tyne
    xllcorner = 356000.0
    yllcorner = 535000.0
    # Wye
    #    xllcorner = 278000.0
    #    yllcorner = 206000.0

    events = read_single_events()

    dlt = 50000.0

    gauge_locations = gauge_locations - [xllcorner - dlt, yllcorner - dlt]
    ixy = np.rint(gauge_locations / 1000.0).astype(int)
    # Reference grid indices
    v_pars = read_varios()
    path = os.path.join(os.path.dirname(__file__), 'outputs', 'S_Norms.csv')
    gau_fil = open("".join(path), 'w')

    for i_sim in range(0, 1, 1):
        ie = 0
        ev = events[ie, 1].astype(int)
        ho = events[ie, 2].astype(int)
        nap = events[ie, 3].astype(int)

        # Do for all selected events
        lin_num = 0
        for lk1 in range(0, lb, 24):
            if int(rain[lk1, 0]) > 1998:
                lin_num = lin_num + 1
                if ev == rain[lk1, 0] and ho == rain[lk1, 1] and nap == rain[lk1, 2]:
                    g_val = np.empty([gauge_locations])
                    #                print(bab)
                    r_act = rain[lk1:lk1 + 24, 4:]
                    rsum = np.sum(r_act, 0)
                    rpos = rsum[rsum >= 0]
                    npos = rpos.shape[0]
                    rda = np.average(rpos)
                    print(rda, rpos)
                    # Find non-parametric density of log precipitation
                    # Using empirical formula for kernel width

                    if np.min(rpos) > 0:
                        d_pos = np.log(rpos)
                        dw = np.std(d_pos)
                        hw = 0.9 * dw / npos ** 0.2
                        #                    print(d_pos)
                        #                    print(dw, hw)
                        p_dens, dens = table_np_gauss(d_pos, hw)

                    r_sort = rsum.argsort()
                    r_rank = r_sort.argsort()
                    prank = r_rank - gauge_locations + npos + 1
                    prank = prank / (npos + 1)

                    k = 0
                    gxy = []
                    for j in range(gauge_locations):
                        if prank[j] > 0:
                            #                        g_val[k] = ndtri(prank[j])
                            g_val[k] = ndtri(p_dens[k])
                            gxy = np.append(gxy, ixy[j])
                            #                        gxy[k] = ixy[j]
                            k = k + 1

                    gxy = np.reshape(gxy, [npos, 2])
                    gxy = gxy.astype(int)

                    # Find monthly rank correlation function
                    ccm = '1.0 Sph(%0.1f)' % (v_pars[ho - 1, 1] / 1000.)

                    sp_sim = spectral_sim.SpectralRandomField(domain_size=(250, 250), covariance_model=ccm)
                    nf = npos * 30

                    sim = []
                    n_prev = npos * 3
                    norm_sm = 99.9
                    for i in range(nf):
                        sim.append(sp_sim.new_simulation())
                    while (norm_sm > 0.1) and (n_prev < nf):
                        n_prev = n_prev + 10
                        aa = make_mat(sim, gxy, n_prev)
                        u, s, v = np.linalg.svd(aa)
                        sv = np.zeros([u.shape[0], v.shape[0]], float)
                        sr = np.zeros([v.shape[0], u.shape[0]], float)
                        for i in range(s.shape[0]):
                            sv[i, i] = s[i]
                            if s[i] > 0.01:
                                sr[i, i] = 1. / s[i]
                        hh = np.dot(v.T, np.dot(sr, u.T))
                        #  zz is the target
                        if gauge_locations == npos:
                            zz = np.copy(g_val)
                        else:
                            zz = g_val[:-gauge_locations + npos]
                        c1 = np.dot(hh.T, zz)
                        c1 = c1[:, None, None]
                        norm_sm = np.sum(c1 ** 2)

                        sma = sim[:n_prev] * c1
                        # Smooth field
                        sm_field = np.sum(sma, axis=0)

                        # Save low norm solution

                    path = os.path.join(os.path.dirname(__file__), 'outputs', 'Smooth_')
                    path = path + '_%i.csv' % lin_num

                    np.savetxt(path, sm_field, fmt='%0.4f', delimiter=',', newline='\n')
                    # Generate kk homogeneous fields

                    kk = 2

                    for ii in range(kk):

                        nf = 50
                        sim = []
                        for i in range(nf):
                            sim.append(sp_sim.new_simulation())
                        aa = make_mat(sim, gxy, nf - 5)
                        u, s, v = np.linalg.svd(aa)
                        sv = np.zeros([u.shape[0], v.shape[0]], float)
                        sr = np.zeros([v.shape[0], u.shape[0]], float)
                        for i in range(s.shape[0]):
                            sv[i, i] = s[i]
                            if s[i] > 0.01:
                                sr[i, i] = 1. / s[i]
                        hh = np.dot(v.T, np.dot(sr, u.T))
                        #  zz is the target
                        zz = sim[nf - 1][gxy[:, 0], gxy[:, 1]]
                        c1 = np.dot(hh.T, zz)
                        norm = np.sum(c1 ** 2) + 1.0
                        c1 = c1[:, None, None]
                        sua = sim[:nf - 5] * c1
                        smb = np.sum(sua, axis=0)

                        smb = (smb - sim[nf - 1]) / norm ** 0.5
                        path = os.path.join(os.path.dirname(__file__), 'outputs', 'Noise_%i' % ii)
                        path = path + '_%i.csv' % lin_num
                        np.savetxt(path, smb, fmt='%0.4f', delimiter=',', newline='\n')
                        nh = "%i" % lin_num + "," + "%0.5f" % norm_sm + "\n"
                        gau_fil.write(nh)

                    ie = ie + 1
                    if ie <= events.shape[0]:
                        ev = events[ie, 1].astype(int)
                        ho = events[ie, 2].astype(int)
                        nap = events[ie, 3].astype(int)
                    else:
                        ev = 9999
                    print(nap, ho, ev, ie, events.shape[0])


def regular_mixed_step():

    gauge_locations = read_gauge_locations()
    number_of_gauges = gauge_locations.shape[0]

    lb, rain = read_rain()
    #  Tyne
    xllcorner = 356000.0
    yllcorner = 535000.0
    # Wye
    #    xllcorner = 278000.0
    #    yllcorner = 206000.0

    events = read_single_events()
    b_events = read_b_s_events()

    dlt = 50000.0

    gauge_locations = gauge_locations - [xllcorner - dlt, yllcorner - dlt]
    ixy = np.rint(gauge_locations / 1000.0).astype(int)

    # Reference grid indices
    v_pars = read_varios()

    for i_sim in range(0, 1, 1):
        ie = 0
        ev = events[ie, 1].astype(int)
        ho = events[ie, 2].astype(int)
        nap = events[ie, 3].astype(int)
        print(ev, ho, nap)

        # Do for all selected events
        lin_num = 0
        for lk1 in range(0, lb, 24):
            if int(rain[lk1, 0]) > 1998:
                lin_num = lin_num + 1
                if ev == rain[lk1, 0] and ho == rain[lk1, 1] and nap == rain[lk1, 2]:
                    ii = 6
                    j1 = b_events[ie, 0]
                    g_val = np.empty([number_of_gauges])
                    r_act = rain[lk1:lk1 + 24, 4:]
                    r_sum = np.sum(r_act, 0)
                    r_pos = r_sum[r_sum >= 0]
                    n_pos = r_pos.shape[0]
                    rda = np.average(r_pos)
                    print(rda, r_pos)
    # Find non-parametric density of log precipitation
    # Using empirical formula for kernel width
                    if np.min(r_pos) > 0:
                        d_pos = np.log(r_pos)
                        dw = np.std(d_pos)
                        hw = 0.9 * dw / n_pos ** 0.2
                        p_dens, dens = table_np_gauss(d_pos, hw)
                    r_sort = r_sum.argsort()
                    r_rank = r_sort.argsort()
                    prank = r_rank - number_of_gauges + n_pos + 1
                    prank = prank / (n_pos + 1)

                    k = 0
                    gxy = []
                    for j in range(number_of_gauges):
                        if prank[j] > 0:
                            g_val[k] = ndtri(p_dens[k])
                            gxy = np.append(gxy, ixy[j])
                            k = k + 1
                    gxy = np.reshape(gxy, [n_pos, 2])
                    gxy = gxy.astype(int)

                    ccm = '1.0 Sph(%0.1f)' % (v_pars[ho - 1, 1] / 1000.)

                    sp_sim = spectral_sim.SpectralRandomField(domain_size=(250, 250), covariance_model=ccm)

                    # Load homogeneous fields
                    path = os.path.join(os.path.dirname(__file__), 'outputs', 'Noise_%i' % ii)
                    path = path + '_%i.csv' % lin_num
                    smb0 = np.loadtxt(path, delimiter=',')

                    ii = ii + 1
                    path = os.path.join(os.path.dirname(__file__), 'outputs', 'Noise_%i' % ii)
                    path = path + '_%i.csv' % lin_num
                    smb1 = np.loadtxt(path, delimiter=',')

                    fi = 2.0 * 3.14159265359 * j1 / 100.0
                    print(smb0.shape, smb1.shape, fi.shape)
                    smc = np.cos(fi) * smb0 + np.sin(fi) * smb1
                    ii = ii + 1
                    path = os.path.join(os.path.dirname(__file__), 'outputs', 'Noise_%i' % ii)
                    path = path + '_%i.csv' % lin_num
                    np.savetxt(path, smc, fmt='%0.4f', delimiter=',', newline='\n')

                    nf = 50
                    sim = []
                    for i in range(nf):
                        sim.append(sp_sim.new_simulation())
                    aa = make_mat(sim, gxy, nf - 5)
                    u, s, v = np.linalg.svd(aa)
                    sv = np.zeros([u.shape[0], v.shape[0]], float)
                    sr = np.zeros([v.shape[0], u.shape[0]], float)
                    for i in range(s.shape[0]):
                        sv[i, i] = s[i]
                        if s[i] > 0.01:
                            sr[i, i] = 1. / s[i]
                    hh = np.dot(v.T, np.dot(sr, u.T))
                    #  zz is the target
                    zz = sim[nf - 1][gxy[:, 0], gxy[:, 1]]
                    c1 = np.dot(hh.T, zz)
                    norm = np.sum(c1 ** 2) + 1.0
                    c1 = c1[:, None, None]
                    sua = sim[:nf - 5] * c1
                    smb = np.sum(sua, axis=0)

                    smb = (smb - sim[nf - 1]) / norm ** 0.5
                    ii = ii + 1
                    path = os.path.join(os.path.dirname(__file__), 'outputs', 'Noise_%i' % ii)
                    path = path + '_%i.csv' % lin_num
                    np.savetxt(path, smb, fmt='%0.4f', delimiter=',', newline='\n')

                    ie = ie + 1
                    if ie <= events.shape[0]:
                        ev = events[ie, 1].astype(int)
                        ho = events[ie, 2].astype(int)
                        nap = events[ie, 3].astype(int)
                    else:
                        ev = 9999
                    print(nap, ho, ev, ie, events.shape[0])


def make_rain_series(iser):
    rain = read_rain()
    lb = rain.shape[0]
    events = read_events()
    grid, grid_size = make_grid_tyne()
    xllcorner = 356000.0
    yllcorner = 535000.0
    dlt = 50000.0

    # Reference grid indices
    xy_area = grid - [xllcorner - dlt, yllcorner - dlt]
    ixy_gr = np.rint(xy_area / 1000.0).astype(int)
    prc_sim = np.empty([len(grid)])

    number_of_events = events.shape[0]

    smooth_file_name = os.path.join(os.path.dirname(__file__), 'outputs', 'S_Norms.csv')
    s_norms = np.loadtxt(smooth_file_name, delimiter=',')

    for i1 in range(10):
        for j1 in range(10):
            ixl = i1*10 + j1
            path = os.path.join(os.path.dirname(__file__), 'outputs', 'Prec_kr_0_24_sim_%i.csv' % ixl)
            son = open("".join(path), 'w')
            path = os.path.join(os.path.dirname(__file__), 'outputs', 'Prec_kr_0_24.csv')
            krn = open("".join(path), 'r')
            lin_one = krn.readline()
            son.write(lin_one)
            lin_num = 0
            is_first = True
            for lk1 in range(0, lb, 24):
                lin = krn.readline()
                lin_num = lin_num + 1
                if int(rain[lk1, 0]) > 1998:
                    is_do = True
                    for i in range(number_of_events):
                        ev = events[i, 1]
                        ho = events[i, 2]
                        nap = events[i, 3]
                        if ev == rain[lk1, 0] and ho == rain[lk1, 1] and nap == rain[lk1, 2]:
                            r_act = rain[lk1:lk1 + 24, 4:]
                            r_sum = np.sum(r_act, 0)
                            r_pos = r_sum[r_sum >= 0]
                            n_pos = r_pos.shape[0]
                            if np.min(r_pos) > 0:
                                d_pos = np.log(r_pos)
                                dw = np.std(d_pos)
                                hw = 0.9 * dw / n_pos ** 0.2
                                p_dens, dens = table_np_gauss(d_pos, hw)
                            else:
                                print("Zero", ev, ho, nap)
                            snum = events[i, 0]+1
                            smooth_file_name = os.path.join(os.path.dirname(__file__), 'outputs', 'Smooth_')
                            smooth_file_name = smooth_file_name + '_%i.csv' % snum
                            g_field = np.loadtxt(smooth_file_name, delimiter=',')
                            noise_file_name = os.path.join(os.path.dirname(__file__), 'outputs', 'Noise_%i' % iser)
                            noise_file_name = noise_file_name + '_%i.csv' % snum
                            noise_field_1 = np.loadtxt(noise_file_name, delimiter=',')
                            noise_new = os.path.join(os.path.dirname(__file__), 'outputs', 'Noise_%i' % (iser+1))
                            noise_new = noise_new + '_%i.csv' % snum
                            noise_field_2 = np.loadtxt(noise_new, delimiter=',')
                            if is_first:
                                fi = 2.0 * 3.14159265359 * i1 / 10.0
                                smc = np.cos(fi) * noise_field_1 + np.sin(fi) * noise_field_2
                                is_first = False
                                print(iser, fi, is_first)
                            else:
                                fi = 2.0 * 3.14159265359 * j1 / 10.0
                                smc = np.cos(fi) * noise_field_1 + np.sin(fi) * noise_field_2
                                is_first = True
                                print(iser, fi)
                            norm_sm = s_norms[i*2, 1]
                            n_field = g_field + ((1. - norm_sm) ** 0.5) * smc
                            p_field = sct.norm.cdf(x=n_field, loc=0, scale=1)
                            ph = p_field * 2000
                            ph = ph.astype(int)
                            ph[ph > 1999] = 1999
                            prc_field = dens[ph]
                            nh = ""
                            for ik in range(len(ixy_gr)):
                                prc_sim[ik] = five_by_five(prc_field, ixy_gr[ik, 0], ixy_gr[ik, 1]) * 0.1
                                nh = nh + "%0.2f" % prc_sim[ik] + ","
                            nh = nh[0:len(nh) - 1] + "\n"
                            son.write(nh)
                            is_do = False
                    if is_do:
                        son.write(lin)


def make_single_rain_series(iser):
    lb, rain = read_rain()
    events = read_single_events()
    grid, grid_size = make_grid_tyne()
    xll_corner = 356000.0
    yll_corner = 535000.0
    dlt = 50000.0

    # Reference grid indices
    xy_area = grid - [xll_corner - dlt, yll_corner - dlt]
    ixy_gr = np.rint(xy_area / 1000.0).astype(int)
    prc_sim = np.empty([len(grid)])

    number_of_events = events.shape[0]

    smooth_file_name = os.path.join(os.path.dirname(__file__), 'outputs', 'S_Norms.csv')
    smooth_fields = np.loadtxt(smooth_file_name, delimiter=',')

    for i1 in range(100):
        ix1 = i1 + 0
        path = os.path.join(os.path.dirname(__file__), 'outputs', 'Prec_kr_0_24_sim_%i.csv' % ix1)
        son = open("".join(path), 'w')
        path = os.path.join(os.path.dirname(__file__), 'outputs', 'Prec_kr_0_24_sim_tst.csv')
        krn = open("".join(path), 'r')
        lin_one = krn.readline()
        son.write(lin_one)
        lin_num = 0
        for lk1 in range(0, lb, 24):
            lin = krn.readline()
            lin_num = lin_num + 1
            if int(rain[lk1, 0]) > 1998:
                is_do = True
                for i in range(number_of_events):
                    ev = events[i, 1]
                    ho = events[i, 2]
                    nap = events[i, 3]
                    if ev == rain[lk1, 0] and ho == rain[lk1, 1] and nap == rain[lk1, 2]:
                        r_act = rain[lk1:lk1 + 24, 4:]
                        r_sum = np.sum(r_act, 0)
                        r_pos = r_sum[r_sum >= 0]
                        n_pos = r_pos.shape[0]
                        if np.min(r_pos) > 0:
                            d_pos = np.log(r_pos)
                            dw = np.std(d_pos)
                            hw = 0.9 * dw / n_pos ** 0.2
                            p_dens, dens = table_np_gauss(d_pos, hw)
                        else:

                            print("Zero", ev, ho, nap)
                            # Read previous best solution
                        s_num = events[i, 0] + 1
                        smooth_file_name = os.path.join(os.path.dirname(__file__), 'outputs', 'Smooth_')
                        smooth_file_name = smooth_file_name + '_%i.csv' % s_num
                        g_field = np.loadtxt(smooth_file_name, delimiter=',')
                        noise_path = os.path.join(os.path.dirname(__file__), 'outputs', 'Noise_%i' % iser)
                        noise_path = noise_path + '_%i.csv' % s_num
                        noise_field_1 = np.loadtxt(noise_path, delimiter=',')
                        noise_new = os.path.join(os.path.dirname(__file__), 'outputs', 'Noise_%i' % (iser + 1))
                        noise_new = noise_new + '_%i.csv' % s_num
                        noise_field2 = np.loadtxt(noise_new, delimiter=',')

                        fi = 2.0 * 3.14159265359 * ix1 / 100.0
                        smc = np.cos(fi) * noise_field_1 + np.sin(fi) * noise_field2
                        norm_sm = smooth_fields[i, 1]
                        n_field = g_field + ((1. - norm_sm) ** 0.5) * smc
                        p_field = sct.norm.cdf(x=n_field, loc=0, scale=1)
                        ph = p_field * 2000
                        ph = ph.astype(int)
                        ph[ph > 1999] = 1999
                        prc_field = dens[ph]
                        nh = ""
                        for ik in range(len(ixy_gr)):
                            prc_sim[ik] = five_by_five(prc_field, ixy_gr[ik, 0],
                                                       ixy_gr[ik, 1]) * 0.1
                            nh = nh + "%0.2f" % prc_sim[ik] + ","
                        nh = nh[0:len(nh) - 1] + "\n"
                        son.write(nh)
                        is_do = False
                if is_do:
                    son.write(lin)


if __name__ == '__main__':
    # Regular_step()
    # r_first()
    iser = 10
    make_rain_series(iser)
    # First_Mixed_step()
    # iser = 8
    # Make_Single_RainSeries(iser)
    # Regular_Mixed_step()
