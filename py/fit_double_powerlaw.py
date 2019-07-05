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
import astropy.modeling.powerlaws as pllaws
from scipy.signal import medfilt
import emcee
import corner
from util import *
import numba

def power_law(pars, t):
    # Unpack pars
    a, xc, k1, k2 = pars
    # Evaluate model
    broken_powerlaw = pllaws.BrokenPowerLaw1D.evaluate(t, a, xc, k1, k2)
    # return 10**(a) * t ** k
    return  broken_powerlaw



def residual(pars, t, data=None, error=None):
    """
    Objective function which calculates the residuals. Using for minimizing.
    """

    # Unpack parameter values
    if type(pars) is list or isinstance(pars, np.ndarray):
        amp_pow, break_pos, slope1_pow, slope2_pow = pars[0], pars[1], pars[2], pars[3]
    else:
        v = pars.valuesdict()
        amp_pow, break_pos, slope1_pow, slope2_pow = v["amp_pow"], v["break_pos"], v["slope1_pow"], v["slope2_pow"]

    # Construct model components
    power_law_component = power_law([amp_pow, break_pos, slope1_pow, slope2_pow], t)

    # Make model
    model = power_law_component

    if data is None:
        return model
    if error is None:
        return (model - data)
    return (model - data)/error



@numba.jit(parallel=True, nopython=False)
def add_powerlaws(pars, x):
    amp_pows = pars["amp_pow"]
    break_poss = pars["break_pos"]
    slope1_pows = pars["slope1_pow"]
    slope2_pows = pars["slope2_pow"]


    out_arr = np.zeros((len(amp_pows), len(x)))
    for ii in numba.prange(len(amp_pows)):
      out_arr[ii, :] = power_law([amp_pows[ii], break_poss[ii], slope1_pows[ii], slope2_pows[ii]], x)

    return out_arr


def main():



    breaks = np.array([14.3076, 14.5415, 14.5760, 14.5889, 14.6051, 14.4015, 14.5930])
    breaks_e = np.array([0.0002, 0.0005, 0.0004, 0.0003, 0.0005, 0.0001, 0.0003])

    slope1s = np.array([0.0006, 0.0317, 0.0475, 0.0116, 0.0297, 0.0458])
    slope1s_e = np.array([0.0001, 0.0002, 0.0001, 0.0001, 0.0001, 0.0001])

    slope2s = np.array([0.2105, 0.1538, 0.2799, 0.2245, 0.1853, 0.2244, 0.2043])
    slope2s_e = np.array([0.0001, 0.0003, 0.0005, 0.0004, 0.0005, 0.0001, 0.0004])

    print(np.mean(breaks), np.sqrt(np.std(breaks)**2 + np.sum(breaks_e**2)))
    print(np.mean(slope1s), np.sqrt(np.std(slope1s)**2 + np.sum(slope1s_e**2)))
    print(np.mean(slope2s), np.sqrt(np.std(slope2s)**2 + np.sum(slope2s_e**2)))

    # slope1_e = np.array([0.001398, 0.001097, 0.001361, 0.000996, 0.001060, 0.001053, 0.001044])
    # slope1s = np.array([0.273295, 0.118976, 0.135172, 0.152629, 0.078017, 0.249216, 0.141438])
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

        # print(logx)
        # exit()


        p = lmfit.Parameters()
        #           (Name,  Value,  Vary,   Min,  Max,  Expr)
        p.add_many(('amp_pow',            -40, True, -np.inf, np.inf),
                   ('break_pos',       14.5, True, 14, 15),
                   ('slope1_pow',       0.27, True, 0, 1),
                   ('slope2_pow',       0.27, True, 0, 1))

        mi = lmfit.minimize(residual, p, method='leastsq', args=(logx, logy, logyerr))
        print(lmfit.report_fit(mi))
        exit()
        # print(lmfit.report_fit(mi.params))
        # ax1.plot(logx, residual(mi.params, logx), lw = 3, zorder=2)
        # pl.ylim((-1e-18, 1e-16))
        # pl.show()
        # exit()



        def lnprob(pars):
            """
            This is the log-likelihood probability for the sampling.
            """
            model = residual(pars, logx)
            return -0.5 * np.sum(((model - logy) / logyerr)**2 + np.log(2 * np.pi * logyerr**2))

        mini = lmfit.Minimizer(lnprob, mi.params)

        nwalkers = 100
        v = mi.params.valuesdict()
        res = mini.emcee(nwalkers=nwalkers, burn=100, steps=1000, thin=1, ntemps=20, is_weighted=True, params=mi.params, seed=12345)
        print(mini.sampler.thermodynamic_integration_log_evidence())

        for i in ["amp_pow", "break_pos", "slope1_pow", "slope2_pow"]:
            mcmc = np.percentile(res.flatchain[i], [16, 50, 84])
            q = np.diff(mcmc)
            print(mcmc[1], q[0], q[1])



        p_lo, p_hi = 50 - 99.9999999/2, 50 + 99.9999999/2

        out_arr = add_powerlaws(res.flatchain, logx_n)
        # out_arr = out_arr/np.median(out_arr, axis=0)
        ax1.plot(logx_n, np.percentile(out_arr, 50, axis=0), lw = 2, zorder=10, color="firebrick", linestyle="dashed", rasterized=True)
        ax1.fill_between(logx_n, np.percentile(out_arr, p_lo, axis=0), np.percentile(out_arr, p_hi, axis=0), alpha=0.5, zorder=9, color="firebrick", rasterized=True)


        bf = np.percentile(out_arr, 50, axis=0)
        f = interpolate.interp1d(logx_n, bf, bounds_error=False, fill_value=np.nan)

        ax2.errorbar(np.log10(nu.value), np.log10(f_nu.value) - f(np.log10(nu.value)), yerr=[np.log10(f_nu.value) - np.log10(f_nu_el.value),np.log10(f_nu_eh.value) - np.log10(f_nu.value)], fmt=".", color="black", alpha=0.2, rasterized=True)
        ax2.plot(np.log10(NUV_nu.value), medfilt(np.log10(NUV_f_nu.value) - f(np.log10(NUV_nu.value)), 11), zorder = 1, rasterized=True)

        ax2.plot(logx_n, bf - bf, lw = 2, zorder=10, color="firebrick", linestyle="dashed", rasterized=True)
        ax2.fill_between(logx_n, np.percentile(out_arr, p_lo, axis=0) - bf, np.percentile(out_arr, p_hi, axis=0) - bf, alpha=0.5, zorder=9, color="firebrick", rasterized=True)


    # print(out_arr.shape)
    ax1.set_xlim(14, 15.3)
    ax1.set_ylim(-25.99, -25.01)
    ax2.set_ylim(-0.05, 0.10)

    ax2.set_xlabel(r"$\log (\nu/\mathrm{Hz})$")
    # ax1.set_ylabel(r'$\log \mathrm{F}_\nu$ [$\mathrm{erg}~\mathrm{s}^{-1}~\mathrm{cm}^{-2}\mathrm{Hz}^{-1}$]')
    ax1.set_ylabel(r'$\log (F_\nu / \mathrm{erg}~\mathrm{s}^{-1}~\mathrm{cm}^{-2}~\mathrm{\AA}^{-1}$)')

    ax2.set_ylabel(r'$\delta \log F_\nu$')


    # Add wavlength axis

    ax3 = ax1.twiny()

    # get axis limits
    xmin, xmax = ax1.get_xlim()
    ax3.set_xlim((xmin, xmax))

    def co(angs):
        return(np.log10(3e18/(angs)))

    nu_arr = np.array([2000, 3000, 5000, 9000, 15000, 25000])
    ax3.set_xticks(co(nu_arr))
    ax3.set_xticklabels(nu_arr)

    ax3.set_xlabel(r"$ \lambda_{\mathrm{obs}}/\mathrm{\AA}$")

    pl.tight_layout()
    pl.savefig("../figures/power_broken_law.pdf")

    # pl.show()
    pl.clf()

    import corner
    corner.corner(res.flatchain, labels=["k", r"$\nu_\mathrm{break}$", r"$\alpha_{\nu,1}$", r"$\alpha_{\nu,2}$"], quantiles=[0.16, 0.5, 0.84], show_titles=True)
    pl.savefig("../figures/Cornerplot_broken_powerlaw.pdf", clobber=True)
if __name__ == '__main__':
    main()
