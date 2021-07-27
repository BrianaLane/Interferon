#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 16:13:46 2021

@author: Briana
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
import astropy.visualization as av
from astropy.modeling import models, fitting
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
from astropy import coordinates as coords
from astropy.stats import sigma_clipped_stats
from astropy.utils.exceptions import AstropyWarning
import warnings

with warnings.catch_warnings():
   warnings.filterwarnings("ignore")
   from photutils import DAOStarFinder

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from astroquery.sdss import SDSS
from photutils import CircularAperture, aperture_photometry, CircularAnnulus


def plot_subframe(ax, frame, vmin=None, vmax=None, c_map='Greys'):

    norm = av.ImageNormalize(frame, interval=av.ZScaleInterval(),
                             stretch=av.SqrtStretch())

    if (vmin is None) or (vmax is None):
        ax.imshow(frame, cmap=c_map, origin='lower', norm=norm)
    else:
        ax.imshow(frame, cmap=c_map, origin='lower', norm=norm,
                  vmin=vmin, vmax=vmax)

    ax.set_xlabel('RA', fontsize=25)
    ax.set_ylabel('DEC', fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=15)


def plot_frame(frame, wcs=None, vmin=None, vmax=None, c_map='Greys',
               save=False, outfile=None):
    
    if wcs.naxis==3:
        wcs = WCS(wcs.to_header(), naxis=2)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=wcs)

    plot_subframe(ax, frame, vmin=vmin, vmax=vmax, c_map=c_map)

    if save:
        try:
            plt.savefig(outfile)
        except:
            print('Invaild outfile path to save plot')

    plt.show()


def find_stars(frame, star_thres=10., num_bright_stars=10,
               star_fwhm=8.0, plot_sources=False):

    mean, median, std = sigma_clipped_stats(frame, sigma=3.0)
    daofind = DAOStarFinder(fwhm=star_fwhm, threshold=star_thres*std,
                            peakmax=64000, exclude_border=True,
                            brightest=num_bright_stars)

    sources = daofind(frame)
    
    if sources is None:
        return None

    else:
        for col in sources.colnames:
            sources[col].info.format = '%.8g'  # for consistent table output
        sources_df = sources.to_pandas()
    
        if plot_sources:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            plot_subframe(ax, frame)
            for i in range(len(sources_df)):
                aperture = CircularAperture((sources_df.iloc[i]['xcentroid'],
                                             sources_df.iloc[i]['ycentroid']),
                                            r=star_fwhm)
    
                aperture.plot(color='red', lw=2.5)
            plt.show()
    
        return sources_df.copy()


def identify_stars(frame, hdr):

    g_coords = coords.SkyCoord(hdr['RA'], hdr['DEC'])
    sources_tab = SDSS.query_crossid(g_coords,
                                     photoobj_fields=['modelMag_g',
                                                      'modelMag_i'])
    sources_df = sources_tab.to_pandas()

    return sources_df


def measure_star_params(frame, sources_df, plot_star_cutouts=False):

    for s in range(len(sources_df)):
        so = sources_df.iloc[s]
        xp, yp = (int(so['xcentroid']), int(so['ycentroid']))
        # this defines half width/height of square cutout
        off = 10

        # makes sure it is not too close to the edge to create cutout
        if (xp > (off-1)) and (yp > (off-1)):
            # build cutout image
            im = frame[yp-off:yp+off, xp-off:xp+off]
            # create array of x and y coordinates of cutout for the fitter
            y_grid, x_grid = np.mgrid[:np.shape(im)[0], :np.shape(im)[1]]
            # finds center for new coodinates of cutout
            x_cen, y_cen = (xp - (xp-off), yp - (yp-off))

            g = models.Gaussian2D(amplitude=so['peak'], x_mean=x_cen,
                                  y_mean=y_cen, x_stddev=1.0, y_stddev=1.0)

            gfit = fitting.LevMarLSQFitter()

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', AstropyWarning)
                # fit 2D Guassian model to cutout image
                fit_g = gfit(g, x_grid, y_grid, im)

            sd = np.average([fit_g.x_stddev.value, fit_g.y_stddev.value])
            xcen_fit_s, ycen_fit_s = (fit_g.x_mean.value, fit_g.y_mean.value)
            fwhm_fit = 2.*np.sqrt(2.*np.log(2))*sd
            flux_fit = 2*np.pi*fit_g.amplitude.value*fit_g.x_stddev.value*fit_g.y_stddev.value
            mag_fit = -2.5*np.log10(flux_fit)

            sources_df.at[s, 'xcentroid_fit'] = xcen_fit_s-x_cen+xp
            sources_df.at[s, 'ycentroid_fit'] = ycen_fit_s-y_cen+yp
            sources_df.at[s, 'fwhm(pixels)'] = fwhm_fit
            sources_df.at[s, 'flux_fit'] = flux_fit
            sources_df.at[s, 'mag_fit'] = mag_fit

            if plot_star_cutouts:
                plt.imshow(im, cmap='gray', origin='lower')
                plt.scatter(x_cen, y_cen, marker='x', color='white')
                fit_x, fit_y = (xcen_fit_s, ycen_fit_s)
                plt.scatter(fit_x, fit_y, marker='x', color='red')
                aperture = CircularAperture((fit_x, fit_y), r=fwhm_fit)
                aperture.plot(color='red', lw=1.5)
                plt.text(1, 2, 'fwhm:'+str(np.round(fwhm_fit, 2)),
                         fontsize=12, color='white')
                plt.text(1, 4, 'peak:'+str(np.round(fit_g.amplitude.value, 2)),
                         fontsize=12, color='white')
                plt.text(1, 6, 'mag:'+str(np.round(mag_fit, 2)),
                         fontsize=12, color='white')
                plt.title(s)
                plt.show()

        # excludes if too close to edge to make cutout
        else:
            sources_df.at[s, 'xcentroid_fit'] = np.NaN
            sources_df.at[s, 'ycentroid_fit'] = np.NaN
            sources_df.at[s, 'fwhm(pixels)'] = np.NaN
            sources_df.at[s, 'flux_fit'] = np.NaN
            sources_df.at[s, 'mag_fit'] = np.NaN

    return sources_df.copy()
