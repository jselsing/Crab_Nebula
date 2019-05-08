#!/usr/local/anaconda3/envs/py36 python
# -*- coding: utf-8 -*-

# Plotting
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as pl
import seaborn as sns; sns.set_style('ticks')

# Imports
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import interpolate
from astropy.convolution import Gaussian1DKernel, Gaussian2DKernel, convolve
from scipy.signal import medfilt
import astropy.units as u
from astropy.time import Time
from util import *
import lmfit
import astropy.units as u
from scipy.signal import medfilt
import emcee

import pyphot
lib = pyphot.get_library()

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



def round_up_to_odd(f):
    f = int(np.ceil(f))
    return f + 1 if f % 2 == 0 else f

def synpassflux(wl, flux, band):
    # Calculate synthetic magnitudes
    n_bands = len(band)

    filt = np.genfromtxt("/Users/jonatanselsing/github/iPTF16geu/data/passbands/%s"%band)
    lamb_T, T = filt[:,0], filt[:,1]
    f = pyphot.Filter(lamb_T, T, name=band, dtype='photon', unit='Angstrom')
    fluxes = f.get_flux(wl, flux, axis=0)
    synmags = -2.5 * np.log10(fluxes) - f.AB_zero_mag
    cwav = np.mean(lamb_T)
    cwave = (float(max(lamb_T[T > np.percentile(T, 10)] - cwav)), float(cwav - min(lamb_T[T > np.percentile(T, 10)])))
    synmag_flux = ((synmags*u.ABmag).to((u.erg/(u.s * u.cm**2 * u.AA)), u.spectral_density(cwav * u.AA))).value
    return synmag_flux, cwav, cwave, synmags

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


    root_dir = "../data/"
    bin_f = 1

    # OBs = ["OB1", "OB3", "OB4", "OB5", "OB6", "OB7", "OB8"]
    OBs = ["OB9"]
    z = 0#3.07
    colors = sns.color_palette("viridis", len(OBs))
    # colors = sns.color_palette("Blues_r", len(OBs))


    wl_out, flux_out, error_out = 0, 0, 0
    for ii, kk in enumerate(OBs):

        off = 0#(len(OBs) - ii) * 2e-17
        mult = 1.0
        # if kk == "OB9_pip":

        ############################## OB ##############################
        f = fits.open(root_dir + "UVB%s.fits"%kk)
        wl = 10. * f[1].data.field("WAVE").flatten()


        q = f[1].data.field("QUAL").flatten()
        try:
            t = f[1].data.field("TRANS").flatten()
        except:
            t = np.ones_like(q)
        mask_wl = (wl > 3200) & (wl < 5550)
        mask_qual = ~q.astype("bool")
        flux = interpolate.interp1d(wl[mask_qual], f[1].data.field("FLUX").flatten()[mask_qual], bounds_error=False, fill_value=0)

        error = interpolate.interp1d(wl[mask_qual], f[1].data.field("ERR").flatten()[mask_qual], bounds_error=False, fill_value=0)
        wl_plot = wl[mask_wl]
        flux = flux(wl_plot)
        error = error(wl_plot)
        # print(flux/error)

        # pl.plot(wl_plot[::1]/(1+z), off + mult*flux[::1], lw=0.3, color = "black", alpha = 0.2, rasterized = True)
        b_wl, b_f, b_e, b_q = bin_spectrum(wl_plot, flux, error, np.zeros_like(flux).astype("bool"), bin_f)
        wl_out, flux_out, error_out = b_wl, b_f, b_e
        wl_stitch, flux_stitch, error_stitch = wl_out, flux_out, error_out


        max_v, min_v = max(medfilt(flux, 101)), min(medfilt(flux, 101))
        f = fits.open(root_dir + "VIS%s.fits"%kk)

        wl = 10. * f[1].data.field("WAVE").flatten()
        q = f[1].data.field("QUAL").flatten()

        try:
            t = f[1].data.field("TRANS").flatten()
        except:
            t = np.ones_like(wl)


        mask_wl = (wl > 5650) & (wl < 10000)
        mask_qual = ~q.astype("bool")

        flux = interpolate.interp1d(wl[mask_qual], f[1].data.field("FLUX").flatten()[mask_qual], bounds_error=False, fill_value=0)
        error = interpolate.interp1d(wl[mask_qual], f[1].data.field("ERR").flatten()[mask_qual], bounds_error=False, fill_value=0)
        wl_plot = wl[mask_wl]
        flux = 0.95*flux(wl_plot) / t[mask_wl]
        error = 0.95*error(wl_plot) / t[mask_wl]

        # pl.plot(wl_plot[::1]/(1+z), off + mult*flux[::1], lw=0.3, color = "black", alpha = 0.2, rasterized = True)
        b_wl, b_f, b_e, b_q = bin_spectrum(wl_plot, flux, error, np.zeros_like(flux).astype("bool"), bin_f)
        wl_out, flux_out, error_out = b_wl, b_f, b_e
        wl_stitch, flux_stitch, error_stitch = np.concatenate([wl_stitch,wl_out]), np.concatenate([flux_stitch,flux_out]), np.concatenate([error_stitch, error_out])

        max_v, min_v = max(max(medfilt(flux, 101)), max_v), min(min(medfilt(flux, 101)), min_v)

        f = fits.open(root_dir + "NIR%s.fits"%kk)
        wl = 10. * f[1].data.field("WAVE").flatten()
        q = f[1].data.field("QUAL").flatten()

        try:
            t = f[1].data.field("TRANS").flatten()
        except:
            t = np.ones_like(wl)


        mask_wl = (wl > 10500) & (wl < 25000) & (t > 0.75)
        mask_qual = ~q.astype("bool")
        flux = interpolate.interp1d(wl[mask_qual], f[1].data.field("FLUX").flatten()[mask_qual], bounds_error=False, fill_value=0)
        error = interpolate.interp1d(wl[mask_qual], f[1].data.field("ERR").flatten()[mask_qual], bounds_error=False, fill_value=0)
        wl_plot = wl[mask_wl]
        flux = 0.95*flux(wl_plot) / t[mask_wl]
        error = 0.95*error(wl_plot) / t[mask_wl]


        # pl.plot(wl_plot[::1]/(1+z), off + mult*flux[::1], lw=0.3, color = "black", alpha = 0.2, rasterized = True)

        b_wl, b_f, b_e, b_q = bin_spectrum(wl_plot, flux, error, np.zeros_like(flux).astype("bool"), int(bin_f))
        wl_out, flux_out, error_out = b_wl, b_f, b_e
        wl_stitch, flux_stitch, error_stitch = np.concatenate([wl_stitch, wl_out]), np.concatenate([flux_stitch,flux_out]), np.concatenate([error_stitch, error_out])



        max_v, min_v = max(max(medfilt(flux, 101)), max_v), min(min(medfilt(flux, 101)), min_v)
        max_std = np.nanstd(flux)
        # print(max_std)
        pl.axvspan(5550, 5650, color="grey", alpha=0.2)
        pl.axvspan(10000, 10500, color="grey", alpha=0.2)
        pl.axvspan(12600, 12800, color="grey", alpha=0.2)
        pl.axvspan(13500, 14500, color="grey", alpha=0.2)
        pl.axvspan(18000, 19500, color="grey", alpha=0.2)
        # print(max(f[0].header['HIERARCH ESO TEL AMBI FWHM END'], f[0].header['HIERARCH ESO TEL AMBI FWHM START']))
        # print(np.diff(wl_out), np.median(np.diff(wl_out)))
        # np.savetxt("UVBVIS_bin30OB3.dat", list(zip(wl_out, flux_out, error_out)))






        # pl.ylim(1.1 * min_v, 1.2 * max_v)
        # pl.ylim(-1e-18, 1.2 * max_v)
        wl = np.arange(min(wl_stitch), max(wl_stitch), np.median(np.diff(wl_stitch)))
        f = interpolate.interp1d(wl_stitch, flux_stitch)
        g = interpolate.interp1d(wl_stitch, error_stitch)

        np.savetxt(root_dir+"%s_stitched.dat"%kk, list(zip(wl, f(wl), g(wl))), fmt='%1.2f %1.4e %1.4e')

        # pl.ylim(-1e-19, 5e-16)
        # pl.ylim(1e-16, 1e-11)

        pl.xlim(2900, 25000)
        # pl.xlabel(r"Observed wavelength  [$\mathrm{\AA}$]")
        pl.ylabel(r'Relative flux density')
        # leg = pl.legend(loc=1)

        # set the linewidth of each legend object
        # for legobj in leg.legendHandles:
        #     legobj.set_linewidth(2.0)
        # pl.semilogy()
        # pl.loglog()




        # pl.axhline(1, linestyle="dashed", color="black", zorder=10)
        # pl.legend()
        # pl.savefig(root_dir + "%s.pdf"%kk)
        # pl.show()

        # pl.clf()


    from dust_extinction.parameter_averages import F04
    # initialize the model
    ext = F04(Rv=3.1)
    import astropy.units as u
    gd71 = np.genfromtxt("../data/fGD71.dat")
    redd = ext.extinguish(gd71[:, 0]*u.angstrom, Ebv=0.2580)
    ext_corr = gd71[:, 1]#/redd
    # pl.plot(gd71[:, 0], ext_corr)
    scaled_flux = flux_stitch/(np.median(flux_stitch[(wl_stitch > 7000) & (wl_stitch < 7100)])/np.median(ext_corr[(gd71[:, 0] > 7000) & (gd71[:, 0] < 7100)]))
    scaled_error = error_stitch/(np.median(flux_stitch[(wl_stitch > 7000) & (wl_stitch < 7100)])/np.median(ext_corr[(gd71[:, 0] > 7000) & (gd71[:, 0] < 7100)]))

    # b_wl, b_f, b_e, b_q = bin_spectrum(wl_plot, flux, error, np.zeros_like(flux).astype("bool"), round_up_to_odd(bin_f/3))
    # wl_out, flux_out, error_out = b_wl, b_f, b_e


    f = interpolate.interp1d(wl_stitch, scaled_flux, bounds_error=False, fill_value=np.nan)
    rat = f(gd71[:, 0])/ext_corr
    h = interpolate.interp1d(wl_stitch, scaled_error, bounds_error=False, fill_value=np.nan)
    rat_e = h(gd71[:, 0])/ext_corr
    # pl.plot(gd71[:, 0], medfilt(rat, 21))


    b_wl, b_rat , b_rate, b_q = bin_spectrum(wl_stitch, rat, rat_e, np.zeros_like(rat).astype("bool"), 5)
    pl.errorbar(b_wl, b_rat, yerr=b_rate, fmt=".", color="black", alpha=0.2, rasterized=True)

    # print(np.nanmean())
    # pl.plot(wl_stitch, medfilt(scaled_flux, 151))

    # pl.show()
    p = lmfit.Parameters()
    #           (Name,  Value,  Vary,   Min,  Max,  Expr)
    p.add_many(('amp_pow',            1, True, -np.inf, np.inf),
               ('slope_pow',       0, True, -1, 1))


    mask = ~(np.isnan(b_rat) | np.isnan(b_rate) | np.isnan(b_wl))

    mi = lmfit.minimize(residual, p, method='Nelder', args=(b_wl[mask], b_rat[mask], b_rate[mask]))
    # print(lmfit.report_fit(mi.params))
    # exit()

    pl.plot(b_wl[mask], residual(mi.params, b_wl[mask]), lw = 3, zorder=2, rasterized=True)
    pl.axhline(1, color="firebrick", rasterized=True)

    # pl.ylim((-1e-18, 1e-16))
    # pl.
    # pl.show()
    # exit()
    # out = mle(logx, logy, s_int=True, po=(7.5,-5.5, 0.1), verbose=True)
    # print(out)
    # pl.plot(wl, 10**out[0] * wl**out[1])
    # pl.plot(wl, flux)
    # pl.show()


    # def lnprob(pars):
    #     """
    #     This is the log-likelihood probability for the sampling.
    #     """
    #     model = residual(pars, b_wl[mask])
    #     return -0.5 * np.sum(((model - b_rat[mask] / b_rate[mask]))**2 + np.log(2 * np.pi * b_rate[mask]**2))

    # mini = lmfit.Minimizer(lnprob, mi.params)

    # nwalkers = 10
    # v = mi.params.valuesdict()
    # res = mini.emcee(nwalkers=nwalkers, burn=100, steps=1000, thin=1, params=mi.params, seed=12345)
    # amps = np.array(res.flatchain["amp_pow"])
    # slopes = np.array(res.flatchain["slope_pow"])

    # out_arr = add_powerlaws(amps, slopes, b_wl)
    # # out_arr = out_arr/np.median(out_arr, axis=0)
    # pl.plot(b_wl, np.percentile(out_arr, 50, axis=0), lw = 2, zorder=10, color="firebrick", linestyle="dashed", rasterized=True)
    # pl.fill_between(b_wl, np.percentile(out_arr, 15.9, axis=0), np.percentile(out_arr, 84.2, axis=0), alpha=0.5, zorder=9, color="firebrick", rasterized=True)
    pl.ylim((0.85, 1.15))


    # Add frequency axis
    ax = pl.gca()
    ax2 = ax.twiny()
    # ax2.plot(np.log10((wl_final * u.angstrom).to(u.Hz, equivalencies=u.spectral()).value), 1 + 0*medfilt(average, 1), linestyle="steps-mid")

    # Add wavlength axis
    # ax = ax.twiny()

    # ax3.set_xlabel(r"$\lambda$  [$\mathrm{\AA}$]")

    # get axis limits
    ymin, ymax = ax.get_xlim()
    # apply function and set transformed values to right axis limits
    ax2.set_xlim((((ymin * u.angstrom).to(u.Hz, equivalencies=u.spectral()).value),((ymax * u.angstrom).to(u.Hz, equivalencies=u.spectral()).value)))
    # set an invisible artist to twin axes
    # to prevent falling back to initial values on rescale events

    def format_func(value, tick_number):
        return str(np.around(np.log10(value), 1))
    ax2.xaxis.set_major_formatter(pl.FuncFormatter(format_func))

    ax2.plot([],[])

    # ax2.semilogx()


    ax.set_xlabel(r"Observed wavelength  [$\mathrm{\AA}$]")
    ax2.set_xlabel(r"Logarithmic observed frequency  [$\mathrm{Hz}$]")


    pl.tight_layout()
    pl.savefig("../figures/GD71_fluxtest.pdf")
    pl.show()
    # p_lo, p_hi = 50 - 68.2689492137086/2, 50 + 68.2689492137086/2
    # l, m, h = np.percentile(np.sort(slopes), [p_lo, 50, p_hi])
    # print(l, m, h)
    # print(m, m - l, h - m)
if __name__ == '__main__':
    main()
