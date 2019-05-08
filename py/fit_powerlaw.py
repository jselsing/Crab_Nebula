#!/usr/local/anaconda3/envs/py36 python
# -*- coding: utf-8 -*-

# Plotting
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as pl
import seaborn as sns; sns.set_style('ticks')

import matplotlib as mpl
# from matplotlib.ticker import FormatStrFormatter
params = {
    'axes.labelsize': 16,
    'font.size': 16,
    'legend.fontsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'text.usetex': True
}
mpl.rcParams.update(params)

# Imports
from astropy.io import fits
from scipy import interpolate
import numpy as np
import pandas as pd
import pymc3 as pm
import theano
import lmfit
import astropy.units as u
from scipy.signal import medfilt
import emcee
import corner
from util import *
import numba

def power_law(pars, t):
    a, k = pars
    # return 10**(a) * t ** k
    return  a  + k * t



def residual(pars, t, data=None, error=None):
    """
    Objective function which calculates the residuals. Using for minimizing.
    """

    # Unpack parameter values
    if type(pars) is list or isinstance(pars, np.ndarray):
        amp_pow, slope_pow = pars[0], pars[1]

    else:
        v = pars.valuesdict()
        amp_pow, slope_pow = v["amp_pow"], v["slope_pow"]

    # Construct model components
    power_law_component = power_law([amp_pow, slope_pow], t)

    # Make model
    model = power_law_component

    if data is None:
        return model
    if error is None:
        return (model - data)
    return (model - data)/error



@numba.jit(parallel=True, nopython=True)
def add_powerlaws(amp, slope, x):

    out_arr = np.zeros((len(amp), len(x)))
    for ii in numba.prange(len(amp)):
      out_arr[ii, :] = amp[ii] +  x * slope[ii]

    return out_arr


def main():

    # amp_e = np.array([0.02056, 0.01603, 0.02003, 0.01385, 0.01543, 0.01574, 0.01446])
    # amps = np.array([-29.5058, -27.2362, -27.4669, -27.7244, -26.6309, -29.1573, -27.5589])
    # print(np.mean(amps), np.sqrt(np.std(amps)**2 + np.sum(amp_e**2)))

    # slope_e = np.array([0.001398, 0.001097, 0.001361, 0.000996, 0.001060, 0.001053, 0.001044])
    # slopes = np.array([0.273295, 0.118976, 0.135172, 0.152629, 0.078017, 0.249216, 0.141438])
    # print(np.mean(slopes), np.sqrt(np.std(slopes)**2 + np.sum(slope_e**2)))
    # print(np.std(slopes))
    # exit()
    root_dir = "../data/"
    # OBs = ["OB1_stitched", "OB3_stitched", "OB4_stitched", "OB5_stitched", "OB6_stitched", "OB7_stitched", "OB8_stitched"]
    # OBs = ["OB1_stitched", "OB3_stitched"]
    OBs = ["weighted_spectrum"]
    fig, (ax1, ax2) = pl.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw = {'height_ratios':[3, 1], 'hspace':0})

    amp = []
    slope = []
    for ii in range(len(OBs)):

        data = np.genfromtxt("../data/%s.dat"%OBs[ii])
        wl, flux, error = data[:, 0], data[:, 1], data[:, 2]
        mask = ((wl > 11600) & (wl < 22500)) | ((wl > 5650) & (wl < 9800) ) | ((wl > 3200) & (wl < 5550))
        flux[((wl > 13300) & (wl < 15100)) | ((wl > 17400) & (wl < 20500))] = None
        mask = (mask & ~np.isnan(flux))# & (flux/error > 30)
        # tell_mask = ((wl > 13300) & (wl < 15100)) | ((wl > 17400) & (wl < 20000))
        wl, flux, error = wl[mask], flux[mask], error[mask]

        nu = (wl * u.angstrom).to(u.Hz, equivalencies=u.spectral())
        f_nu = (flux * u.erg * u.cm**-2 * u.s**-1 * u.angstrom**-1).to(u.erg * u.cm**-2 * u.s**-1 * u.Hz**-1, equivalencies=u.spectral_density(wl * u.angstrom))
        # f_nu_e = (error * u.erg / u.cm**2 / u.s / u.angstrom).to(u.erg / u.cm**2 / u.s / u.Hz, equivalencies=u.spectral_density(wl * u.angstrom))
        f_nu_el = ((flux - error) * u.erg * u.cm**-2 * u.s**-1 * u.angstrom**-1).to(u.erg * u.cm**-2 * u.s**-1 * u.Hz**-1, equivalencies=u.spectral_density(wl * u.angstrom))
        f_nu_eh = ((flux + error) * u.erg * u.cm**-2 * u.s**-1 * u.angstrom**-1).to(u.erg * u.cm**-2 * u.s**-1 * u.Hz**-1, equivalencies=u.spectral_density(wl * u.angstrom))
        f_nu_e = np.mean([f_nu.value - f_nu_el.value,f_nu_eh.value - f_nu.value], axis=0)

        ax1.errorbar(np.log10(nu.value), np.log10(f_nu.value), yerr=[np.log10(f_nu.value) - np.log10(f_nu_el.value),np.log10(f_nu_eh.value) - np.log10(f_nu.value)], fmt=".", color="black", alpha=0.2, rasterized=True)
        # pl.plot(np.log10(nu.value), medfilt(np.log10(f_nu_e.value), 21))
        # pl.show()
        # exit()
        NUV = np.genfromtxt("../data/crabNUV.txt")
        NUV_wl, NUV_flux = NUV[50:-20, 0], medfilt(NUV[50:-20, 1], 1)
        dust_ext = correct_for_dust(NUV_wl, 0.52)

        NUV_nu = (NUV_wl * u.angstrom).to(u.Hz, equivalencies=u.spectral())
        NUV_f_nu = (dust_ext*NUV_flux * u.erg * u.cm**-2 * u.s**-1 * u.angstrom**-1).to(u.erg * u.cm**-2 * u.s**-1 * u.Hz**-1, equivalencies=u.spectral_density(NUV_wl * u.angstrom))


        ax1.plot(np.log10(NUV_nu.value), np.log10(NUV_f_nu.value), zorder = 1, rasterized=True)

        # pl.show()

        # pl.show()
        # exit()

        mask = (flux <= 1e-16) #| (flux/error < 10) #| (wl > 10000)
        x, y, yerr = nu.value[~mask][::1], f_nu.value[~mask][::1], f_nu_e[~mask][::1]
        logx = np.log10(x)
        logx_n = np.arange(12, 16, 0.001)[::-1]
        # print(logx, logx_n)
        # exit()
        logy = np.log10(y)
        logyerr = np.log10(np.array(y)+np.array(yerr)) - logy



        p = lmfit.Parameters()
        #           (Name,  Value,  Vary,   Min,  Max,  Expr)
        p.add_many(('amp_pow',            -40, True, -np.inf, np.inf),
                   ('slope_pow',       0.27, True, 0, 1))

        mi = lmfit.minimize(residual, p, method='Nelder', args=(logx, logy, logyerr))
        # print(lmfit.report_fit(mi.params))
        # exit()

        # pl.plot(logx, residual(mi.params, logx), lw = 3, zorder=2)
        # pl.ylim((-1e-18, 1e-16))
        # pl.
        # pl.show()
        # exit()
        # out = mle(logx, logy, s_int=True, po=(7.5,-5.5, 0.1), verbose=True)
        # print(out)
        # pl.plot(wl, 10**out[0] * wl**out[1])
        # pl.plot(wl, flux)
        # pl.show()


        def lnprob(pars):
            """
            This is the log-likelihood probability for the sampling.
            """
            model = residual(pars, logx)
            return -0.5 * np.sum(((model - logy) / logyerr)**2 + np.log(2 * np.pi * logyerr**2))

        mini = lmfit.Minimizer(lnprob, mi.params)

        nwalkers = 10
        v = mi.params.valuesdict()
        res = mini.emcee(nwalkers=nwalkers, burn=500, steps=5000, thin=1, params=mi.params, seed=12345)
        amp = (res.flatchain["amp_pow"])
        slope = (res.flatchain["slope_pow"])
        # res_out = np.conc
        # exit()


        # low, mid, high, names = [], [], [], []
        # for kk in res.params.valuesdict().keys():
        #     names.append(str(kk))

        #     low.append(np.percentile(res.flatchain[str(kk)], [15.9])[0])
        #     mid.append(np.percentile(res.flatchain[str(kk)], [50])[0])
        #     high.append(np.percentile(res.flatchain[str(kk)], [84.2])[0])
        # low, mid, high = np.array(low), np.array(mid), np.array(high)
        # print(mid[1], mid[1] - low[1], high[1] - mid[1])
        # pl.plot(logx_n, residual(mi.params, logx_n), lw = 2, zorder=3, color="firebrick")
        # pl.fill_between(logx_n, residual(low, logx_n), residual(high, logx_n), alpha=0.5, zorder=2)
        # pl.plot(wl, residual(high, wl), lw = 3, linestyle="dashed")
        # pl.xlim(3200, 6000)
        # pl.ylim(-1e-18, 1.5e-16)
        # p_lo, p_hi = 50 - 68.2689492137086/2, 50 + 68.2689492137086/2
        # print(p_lo, p_hi)
        # amps = np.hstack(amp)

        amps = np.random.normal(loc = 0, scale = 0.02, size= int(1e4))
        slopes = np.random.normal(loc = 0.1641, scale = 0.0656, size= int(1e4))
        nu_0 = np.log10(4.283e14)
        spec_0 =  np.nanmedian(logy[(logx > nu_0 - 0.1) & (logx < nu_0 + 0.1)])
        amps = spec_0 - nu_0 * slopes + amps
        # print(amps)
        # slopes = np.hstack(slope)
        # l, m ,h = np.percentile(np.sort(amps), [p_lo, 50, p_hi])
        # print(l, m ,h)
        # print(m, m - l, h - m)
        out_arr = add_powerlaws(amps, slopes, logx_n)
        # out_arr = out_arr/np.median(out_arr, axis=0)
        ax1.plot(logx_n, np.percentile(out_arr, 50, axis=0), lw = 2, zorder=10, color="firebrick", linestyle="dashed", rasterized=True)
        ax1.fill_between(logx_n, np.percentile(out_arr, 15.9, axis=0), np.percentile(out_arr, 84.2, axis=0), alpha=0.5, zorder=9, color="firebrick", rasterized=True)


        bf = np.percentile(out_arr, 50, axis=0)
        f = interpolate.interp1d(logx_n, bf, bounds_error=False, fill_value=np.nan)

        ax2.errorbar(np.log10(nu.value), np.log10(f_nu.value) - f(np.log10(nu.value)), yerr=[np.log10(f_nu.value) - np.log10(f_nu_el.value),np.log10(f_nu_eh.value) - np.log10(f_nu.value)], fmt=".", color="black", alpha=0.2, rasterized=True)
        ax2.plot(np.log10(NUV_nu.value), medfilt(np.log10(NUV_f_nu.value) - f(np.log10(NUV_nu.value)), 11), zorder = 1, rasterized=True)

        ax2.plot(logx_n, bf - bf, lw = 2, zorder=10, color="firebrick", linestyle="dashed", rasterized=True)
        ax2.fill_between(logx_n, np.percentile(out_arr, 15.9, axis=0) - bf, np.percentile(out_arr, 84.2, axis=0) - bf, alpha=0.5, zorder=9, color="firebrick", rasterized=True)


    # print(out_arr.shape)
    ax1.set_xlim(14, 15.3)
    ax1.set_ylim(-25.99, -25.01)
    ax2.set_ylim(-0.05, 0.10)

    ax2.set_xlabel(r"$\log \nu$  [$\mathrm{Hz}$]")
    ax1.set_ylabel(r'$\log \mathrm{F}_\nu$ [$\mathrm{erg}~\mathrm{s}^{-1}~\mathrm{cm}^{-2}\mathrm{Hz}^{-1}$]')
    ax2.set_ylabel(r'$\delta \log \mathrm{F}_\nu$')


    # Add wavlength axis
    ax3 = ax1.twiny()

    ax3.set_xlabel(r"$\lambda$  [$\mathrm{\AA}$]")

    # get axis limits
    ymin, ymax = ax2.get_xlim()
    # apply function and set transformed values to right axis limits
    print((10**(ymin) * u.Hz).to(u.angstrom, equivalencies=u.spectral()),(10**(ymax) * u.Hz).to(u.angstrom, equivalencies=u.spectral()))
    ax3.set_xlim((((ymin) * u.Hz).to(u.angstrom, equivalencies=u.spectral()).value,((ymax) * u.Hz).to(u.angstrom, equivalencies=u.spectral()).value))


    def format_func(value, tick_number):
        return str(np.around(10**(value), 1))
    ax2.xaxis.set_major_formatter(pl.FuncFormatter(format_func))


    # set an invisible artist to twin axes
    # to prevent falling back to initial values on rescale events
    ax3.plot([],[])




    pl.tight_layout()
    pl.savefig("../figures/power_law.pdf")

    pl.show()

if __name__ == '__main__':
    main()
