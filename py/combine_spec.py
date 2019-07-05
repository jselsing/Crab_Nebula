#!/usr/local/anaconda3/envs/py36 python
# -*- coding: utf-8 -*-

# Plotting
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as pl
import seaborn as sns; sns.set_style('ticks')

import matplotlib as mpl
# from matplotlib.ticker import FormatStrFormatter
params = {
    'axes.labelsize': 12,
    'font.size': 12,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': True
}
mpl.rcParams.update(params)


# Imports
from astropy.io import fits
from scipy import interpolate
import numpy as np
import pandas as pd
from util import *
import pyphot
lib = pyphot.get_library()
import astropy.units as u
from scipy.signal import medfilt


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


def main():





    root_dir = "../data/"
    OBs = ["OB1", "OB3", "OB4", "OB5", "OB6", "OB7", "OB8"]

    wl_out, flux_out, error_out = [0]*len(OBs), [0]*len(OBs), [0]*len(OBs)
    offsets = [0.95, 0.95, 0.85, 0.95, 0.95, 0.95, 0.95]
    bin_f = 1
    for ii, kk in enumerate(OBs):

        off = 0
        mult = 1.0

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
        flux = interpolate.interp1d(wl[mask_qual], f[1].data.field("FLUX").flatten()[mask_qual], bounds_error=False, fill_value=np.nan)

        error = interpolate.interp1d(wl[mask_qual], f[1].data.field("ERR").flatten()[mask_qual], bounds_error=False, fill_value=np.nan)
        wl_plot = wl[mask_wl]
        dust_ext = correct_for_dust(wl_plot, 0.52)
        flux = dust_ext*flux(wl_plot)
        error = dust_ext*error(wl_plot)
        b_wl, b_f, b_e, b_q = bin_spectrum(wl_plot, flux, error, np.zeros_like(flux).astype("bool"), bin_f)

        wl_stitch, flux_stitch, error_stitch = b_wl, b_f, b_e



        f = fits.open(root_dir + "VIS%s.fits"%kk)

        wl = 10. * f[1].data.field("WAVE").flatten()
        q = f[1].data.field("QUAL").flatten()

        try:
            t = f[1].data.field("TRANS").flatten()
        except:
            t = np.ones_like(wl)


        mask_wl = (wl > 5650) & (wl < 10000)
        mask_qual = ~q.astype("bool")

        flux = interpolate.interp1d(wl[mask_qual], f[1].data.field("FLUX").flatten()[mask_qual], bounds_error=False, fill_value=np.nan)
        error = interpolate.interp1d(wl[mask_qual], f[1].data.field("ERR").flatten()[mask_qual], bounds_error=False, fill_value=np.nan)
        wl_plot = wl[mask_wl]
        dust_ext = correct_for_dust(wl_plot, 0.52)
        flux = offsets[ii]*dust_ext*flux(wl_plot) / t[mask_wl]
        error = offsets[ii]*dust_ext*error(wl_plot) / t[mask_wl]
        b_wl, b_f, b_e, b_q = bin_spectrum(wl_plot, flux, error, np.zeros_like(flux).astype("bool"), bin_f)

        wl_stitch, flux_stitch, error_stitch = np.concatenate([wl_stitch,b_wl]), np.concatenate([flux_stitch,b_f]), np.concatenate([error_stitch, b_e])
        # wl_stitch, flux_stitch, error_stitch = np.concatenate([wl_stitch,wl_plot]), np.concatenate([flux_stitch,flux]), np.concatenate([error_stitch, error])





        f = fits.open(root_dir + "NIR%s.fits"%kk)
        wl = 10. * f[1].data.field("WAVE").flatten()
        q = f[1].data.field("QUAL").flatten()

        try:
            t = f[1].data.field("TRANS").flatten()
        except:
            t = np.ones_like(wl)


        mask_wl = (wl > 10000) & (wl < 25000)
        mask_qual = ~q.astype("bool")
        flux = interpolate.interp1d(wl[mask_qual], f[1].data.field("FLUX").flatten()[mask_qual], bounds_error=False, fill_value=np.nan)
        error = interpolate.interp1d(wl[mask_qual], f[1].data.field("ERR").flatten()[mask_qual], bounds_error=False, fill_value=np.nan)
        wl_plot = wl[mask_wl]
        dust_ext = correct_for_dust(wl_plot, 0.52)
        flux = offsets[ii]*dust_ext*flux(wl_plot) / t[mask_wl]
        error = offsets[ii]*dust_ext*error(wl_plot) / t[mask_wl]


        b_wl, b_f, b_e, b_q = bin_spectrum(wl_plot, flux, error, np.zeros_like(flux).astype("bool"), bin_f*3)
        wl_stitch, flux_stitch, error_stitch = np.concatenate([wl_stitch,b_wl]), np.concatenate([flux_stitch,b_f]), np.concatenate([error_stitch, b_e])



        # pl.scatter(wl_stitch, flux_stitch)
        # pl.show()
        wl = np.arange(min(wl_stitch), max(wl_stitch), np.median(np.diff(wl_stitch)))
        f = interpolate.interp1d(wl_stitch, flux_stitch, fill_value=np.nan)
        g = interpolate.interp1d(wl_stitch, error_stitch, fill_value=np.nan)
        # h = interpolate.interp1d(wl_stitch, bp_stitch, fill_value=np.nan)

        wl_out[ii] = wl
        flux_out[ii] = f(wl)
        # flux_out[ii][h(wl) != 0] = np.nan
        error_out[ii] = g(wl)
        # error_out[ii][h(wl) != 0] = np.nan
#
        # np.savetxt(root_dir+"%s_stitched.dat"%kk, list(zip(wl, flux_out[ii], error_out[ii])), fmt='%1.2f %1.4e %1.4e')


    min_wl = 100000
    max_wl = 0
    flux_scaling = [0]*len(OBs)
    for ii, kk in enumerate(OBs):
        min_wl_temp = min(wl_out[ii])
        min_wl = min(min_wl_temp, min_wl)
        max_wl_temp = max(wl_out[ii])
        max_wl = max(max_wl_temp, max_wl)
        dwl = np.median(np.diff(wl_out[ii]))
        flux_scaling[ii] =  np.median(flux_out[ii][(wl_out[ii] > 7000) & (wl_out[ii] < 7100)])
        np.savetxt(root_dir+"%s_stitched.dat"%kk, list(zip(wl, flux_out[ii]*(flux_scaling[0]/flux_scaling[ii]), error_out[ii]*(flux_scaling[0]/flux_scaling[ii]))), fmt='%1.2f %1.4e %1.4e')


    # print(flux_scaling)

    # print(min_wl, max_wl, dwl)
    wl_final = np.arange(min_wl, max_wl, dwl)
    crab_spectra_flux = np.zeros((len(OBs), len(wl_final)))
    crab_spectra_error = np.zeros((len(OBs), len(wl_final)))

    for ii, kk in enumerate(OBs):
        mask = np.isnan(flux_out[ii]).astype("int")
        f = interpolate.interp1d(wl_out[ii], flux_out[ii], bounds_error=False, fill_value=np.nan)
        g = interpolate.interp1d(wl_out[ii], error_out[ii], bounds_error=False, fill_value=np.nan)
        crab_spectra_flux[ii, :] = f(wl_final)*(flux_scaling[0]/flux_scaling[ii])
        crab_spectra_error[ii, :] = g(wl_final)*(flux_scaling[0]/flux_scaling[ii])

        pl.plot(wl_final, crab_spectra_flux[ii, :], color="black", alpha = 0.2, linestyle="steps-mid", rasterized=True)


    print(np.mean(flux_scaling), np.std(flux_scaling))
    print(np.mean(flux_scaling[0]/flux_scaling), np.std(flux_scaling[0]/flux_scaling))


    # l, m, h =(np.percentile(flux_scaling[0]/flux_scaling, [15.9, 50, 84.2]))
    # print(m, m - l, h - m)
    # exit()

    # Median spectrum
    med_spec = np.nanmedian(crab_spectra_flux[:, :], axis=0)
    np.savetxt(root_dir+"median_spectrum.dat", list(zip(wl_final, med_spec)), fmt='%1.2f %1.4e')

    # Weighted mean
    weight = 1.0 / (crab_spectra_error ** 2.0)
    average = np.ma.sum(crab_spectra_flux * weight, axis=0) / np.ma.sum(weight, axis=0)
    std = np.sqrt(np.ma.sum(crab_spectra_error**2 * weight**2, axis=0) / np.ma.sum(weight, axis=0)**2)

    # average =  np.nanmean(crab_spectra_flux, axis=0)
    # std = np.sqrt(np.nansum(crab_spectra_error**2, axis=0))
    # print(average)
    # print(std)

    np.savetxt(root_dir+"weighted_spectrum.dat", list(zip(wl_final, average, std)), fmt='%1.2f %1.4e %1.4e')
    # exit()
    # pl.plot(wl_final, med_spec)
    pl.plot(wl_final, medfilt(average, 1), linestyle="steps-mid", rasterized=True)





    # NUV = np.genfromtxt("../data/crabNUV.txt")
    # dust_ext = correct_for_dust(NUV[:, 0], 0.52)
    # pl.plot(NUV[:, 0], dust_ext*medfilt(NUV[:, 1], 11), linestyle="steps-mid")
    # pl.show()


    # for ii, OB in enumerate(OBs):


    # passbands = [
        # "FORS2_U.dat", "FORS2_B.dat", "FORS2_V.dat", "FORS2_R.dat", "FORS2_I.dat", "2MASS_J.dat", "2MASS_H.dat", "2MASS_Ks.dat"
        # ]
    passbands = ["FORS1_U.dat", "FORS1_B.dat", "FORS1_V.dat", "FORS1_R.dat", "FORS1_I.dat", "FORS2_z.dat", "NACO_J.dat", "NACO_H.dat", "NACO_Ks.dat"]
    photometry = pd.read_csv("../data/Crab_phot.csv")

    vega_conv = pd.read_csv("~/github/iPTF16geu/data/passbands/VEGA_AB.csv")
    # cmap1 = sns.color_palette("viridis", len(passbands))
    cmap1 = sns.color_palette("plasma", 5+len(passbands))

    for pp, ss in enumerate(passbands):
        pass_name = ss.split("_")[-1].split(".")[0]
        meas_mag = photometry.loc[photometry['Bandpass'] == pass_name]
        print(pass_name)
        vega = vega_conv.loc[vega_conv["Band"] == pass_name]
        print(meas_mag["CrabPulsar+Knot"], vega["dmag"].values)
        meas_mag["CrabPulsar+Knot"] = meas_mag["CrabPulsar+Knot"] + vega["dmag"].values
        dust_ext = correct_for_dust(vega["leff"].values, 0.52)
        # print(dust_ext, ebv)



        nanmask = (np.isnan(average) | np.isnan(std)) | (np.isinf(average) | np.isinf(std))
        synmag_flux, cwav, cwave, synmag = synpassflux(wl_final[~nanmask], average[~nanmask], ss)
        synmag_error, cwav, cwave, synmag_err = synpassflux(wl_final[~nanmask], std[~nanmask], ss)
        _, _, _, synmag_err_up = synpassflux(wl_final[~nanmask], average[~nanmask] + std[~nanmask], ss)
        _, _, _, synmag_err_low = synpassflux(wl_final[~nanmask], average[~nanmask] - std[~nanmask], ss)
        e_u = (synmag - synmag_err_up)
        e_l = (synmag_err_low - synmag)
        # pl.errorbar(cwav, synmag_flux, xerr = [[cwave]], yerr = synmag_error,  fmt = 'o', color=cmap1[pp], zorder = 10, ms = 5, elinewidth=1.7, label = "%s = %s$^{+%s}_{-%s}$"%(pass_name, np.around(synmag, 2), np.around(e_u, 2), np.around(e_l, 2)))
        # pl.errorbar(cwav, synmag_flux, xerr = [[cwave]], yerr = synmag_error,  fmt = 'o', color=cmap1[pp], zorder = 10, ms = 5, elinewidth=1.7, label = "%s"%(pass_name))

        if meas_mag.shape[0] > 0:
            meas_flux = ((meas_mag["CrabPulsar+Knot"].values*u.ABmag).to((u.erg/(u.s * u.cm**2 * u.AA)), u.spectral_density(cwav * u.AA))).value
            meas_flux_up =  (((meas_mag["CrabPulsar+Knot"].values + meas_mag["CrabPulsar+Knot_e"].values)*u.ABmag).to((u.erg/(u.s * u.cm**2 * u.AA)), u.spectral_density(cwav * u.AA))).value
            meas_flux_do =  (((meas_mag["CrabPulsar+Knot"].values - meas_mag["CrabPulsar+Knot_e"].values)*u.ABmag).to((u.erg/(u.s * u.cm**2 * u.AA)), u.spectral_density(cwav * u.AA))).value

            pl.errorbar(cwav, dust_ext*meas_flux, xerr = [[cwave]], yerr = [[(meas_flux_do, meas_flux_up)]],  fmt = 'o', color=cmap1[pp], ms = 7, zorder=9, label = "%s"%(pass_name))
    pl.xlim(2900, 25000)
    pl.ylim(1e-16, 2e-14)






    pl.ylabel(r'$\log (F_\lambda / \mathrm{erg}~\mathrm{s}^{-1}~\mathrm{cm}^{-2}~\mathrm{\AA}^{-1}$)')
    pl.semilogy()
    pl.legend()

    # Add frequency axis
    ax = pl.gca()
    ax2 = pl.twiny()

    # get axis limits
    xmin, xmax = ax.get_xlim()
   ax2.set_xlim((xmin, xmax))

    def co(angs):
        return(3e18/(10**angs))
    nu_arr = np.array([15, 14.6, 14.4, 14.3, 14.2, 14.1])
    ax2.set_xticks(co(nu_arr))
    ax2.set_xticklabels(nu_arr)



    ax.set_xlabel(r"$ \lambda_{\mathrm{obs}}/\mathrm{\AA}$")
    ax2.set_xlabel(r"$\log (\nu/\mathrm{Hz})$")

    pl.tight_layout()
    pl.savefig("../figures/combined_spectrum.pdf")
    pl.show()


if __name__ == '__main__':
    main()
