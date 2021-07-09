#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 15:28:36 2021

@author: Briana
"""

import os.path as op
import pandas as pd
import numpy as np
import glob
import datetime as dt
import matplotlib.pyplot as plt
import warnings

from astropy import coordinates as coords
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.modeling import models, fitting
import astropy.visualization as av
from astropy.utils.exceptions import AstropyWarning

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    from photutils import DAOStarFinder
    
from astroquery.sdss import SDSS
from photutils import CircularAperture, aperture_photometry, CircularAnnulus


class guider_observations():
    def __init__(self, guider_path, guider_as_per_pix=0.51):
        
        print('BUILD guider observation: [GUIDER]')

        self.guider_path = guider_path
        self.guider_ps = guider_as_per_pix
        guider_files = glob.glob(op.join(guider_path, '*.fits'))

        self.guider_df = pd.DataFrame({'filename': guider_files,
                                       'obs_datetime': np.NaN,
                                       'exptime(s)': np.NaN})

        for g in range(len(self.guider_df)):
            hdu = fits.open(self.guider_df.iloc[g]['filename'])
            hdr_g = hdu[0].header
            date_hdr = hdr_g['DATE-OBS']
            time_hdr = hdr_g['UT']
            obs_dt = dt.datetime.strptime(date_hdr+'T'+time_hdr,
                                          '%Y-%m-%dT%H:%M:%S.%f')
            self.guider_df.at[g, 'obs_datetime'] = obs_dt
            self.guider_df.at[g, 'exptime(s)'] = hdr_g['EXPTIME']
            hdu.close()
  
    # guider_name (int/str): can be either guider index in guider_df or filename
    def inspect_guider_frame(self, guider_name, vmin=None, vmax=None):
        if isinstance(guider_name, int):
            if guider_name < len(self.guider_df):
                self.guider_df.iloc[guider_name]['filename']
                hdu = fits.open(self.guider_df.iloc[guider_name]['filename'])
            else:
                print('Guider index out of bounds')
                return None
        elif isinstance(guider_name, str):
            if guider_name[-5::] == '.fits':
                hdu = fits.open(guider_name)
            else:
                print('Invalid guider filename, must be fits file or \
                      guider index integer')
                return None
        else:
            print('Need to provide guider path/filename.fits (str) or index \
                  (int) of guider frame in guider_df')
            return None

        hdr = hdu[0].header
        dat = hdu[0].data
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyWarning)
            im_wcs = WCS(hdr)
        hdu.close()

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1, projection=im_wcs)

        norm = av.ImageNormalize(dat, interval=av.ZScaleInterval(),
                                 stretch=av.SqrtStretch())

        if (vmin is None) or (vmax is None):
            ax.imshow(dat, cmap='Greys', norm=norm)
        else:
            ax.imshow(dat, cmap='Greys', norm=norm, vmin=vmin, vmax=vmax)

        ax.set_ylabel('RA', fontsize=25)
        ax.set_xlabel('DEC', fontsize=25)
        ax.tick_params(axis='both', which='major', labelsize=15)
        plt.show()

    def find_guide_stars(self, guide_ind, star_thres=10., num_bright_stars=10,
                         star_fwhm=8.0, plot_guide_frame=False):

        hdu = fits.open(self.guider_df.iloc[guide_ind]['filename'])
        g_dat = hdu[0].data

        mean, median, std = sigma_clipped_stats(g_dat, sigma=3.0)
        daofind = DAOStarFinder(fwhm=star_fwhm, threshold=star_thres*std,
                                peakmax=64000, exclude_border=True,
                                brightest=num_bright_stars)

        sources = daofind(g_dat)
        for col in sources.colnames:
            sources[col].info.format = '%.8g'  # for consistent table output
        sources_df = sources.to_pandas()

        if plot_guide_frame:
            fig, ax = plt.subplots(1,1,figsize=(10, 10))
            ax.imshow(g_dat, cmap='gray', vmin=1310, vmax=1400, origin='lower')
            for i in range(len(sources_df)):
                aperture = CircularAperture((sources_df.iloc[i]['xcentroid'],
                                             sources_df.iloc[i]['ycentroid']),r=star_fwhm)

                aperture.plot(color='red', lw=2.5)
            plt.show()

        hdu.close()

        return sources_df.copy()

    def identify_stars(self, guide_ind):
        gframe_df = self.guider_df.iloc[guide_ind]
        hdu = fits.open(gframe_df['filename'])
        dat = hdu[0].data
        hdr = hdu[0].header

        g_coords = coords.SkyCoord(hdr['RA'], hdr['DEC'])
        sources_tab = SDSS.query_crossid(g_coords, photoobj_fields=['modelMag_g', 'modelMag_i'])
        sources_df = soures_tab.to_pandas()
        hdu.close()
        return sources_df

    def measure_guide_star_params(self, guide_ind, sources_df,
                                  plot_star_cutouts=False):

        hdu = fits.open(self.guider_df.iloc[guide_ind]['filename'])
        g_dat = hdu[0].data

        # divide guider image by exptime to get counts per second
        guide_et = self.guider_df.iloc[guide_ind]['exptime(s)']
        g_dat_perS = g_dat/guide_et #counts per second

        # remove background (approx bias) level from guider image
        mean, median, std = sigma_clipped_stats(g_dat_perS, sigma=3.0)
        g_dat_perS_noBKGD = g_dat_perS - median

        for s in range(len(sources_df)):
            so = sources_df.iloc[s]
            xp, yp = (int(so['xcentroid']), int(so['ycentroid']))
            # this defines half width/height of square cutout
            off = 10

            # makes sure it is not too close to the edge to create cutout
            if (xp > (off-1)) and (yp > (off-1)):
                # build cutout image 
                im = g_dat_perS_noBKGD[yp-off:yp+off, xp-off:xp+off]
                # create array of x and y coordinates of cutout for the fitter
                y_grid, x_grid = np.mgrid[:np.shape(im)[0], :np.shape(im)[1]]
                # finds center for new coodinates of cutout
                x_cen, y_cen = (xp - (xp-off), yp - (yp-off))

                g = models.Gaussian2D(amplitude=so['peak']-mean, x_mean=x_cen,
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
                sources_df.at[s, 'fwhm(arcseconds)'] = fwhm_fit*self.guider_ps
                sources_df.at[s, 'flux_fit'] = flux_fit
                sources_df.at[s, 'mag_fit'] = mag_fit

                if plot_star_cutouts:
                    plt.imshow(im,cmap='gray',origin='lower')
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
                sources_df.at[s, 'fwhm(arcseconds)'] = np.NaN
                sources_df.at[s, 'flux_fit'] = np.NaN
                sources_df.at[s, 'mag_fit'] = np.NaN

        hdu.close()
        return sources_df.copy()

    def flag_stars(self, sources_df, fwhm_lim=(0.5, 10), mag_lim=10):
        for i in range(len(sources_df)):
            if (np.isnan(sources_df.iloc[i]['mag_fit'])) or (np.isnan(sources_df.iloc[i]['mag_fit'])):
                sources_df.at[i, 'bad_flag'] = True
            elif (sources_df.iloc[i]['fwhm(arcseconds)'] > fwhm_lim[1]) or (sources_df.iloc[i]['fwhm(arcseconds)'] < fwhm_lim[0]):
                sources_df.at[i, 'bad_flag'] = True
            elif (sources_df.iloc[i]['mag_fit'] > mag_lim):
                sources_df.at[i, 'bad_flag'] = True
            else:
                sources_df.at[i, 'bad_flag'] = False

        return sources_df.copy()

    def find_ref_guide_frame(self, guide_ind_list, star_thres=10.,
                             num_bright_stars=10, star_fwhm=8.0,
                             fwhm_lim=(0.5, 10), mag_lim=10):

        for g in range(len(guide_ind_list)):
            guide_ind = guide_ind_list[g]

            source_ex = self.find_guide_stars(guide_ind, star_thres=star_thres,
                                              num_bright_stars=num_bright_stars,
                                              star_fwhm=star_fwhm)
            if len(source_ex)>1:
                source_fit = self.measure_guide_star_params(guide_ind,
                                                            source_ex)
                source_fit = self.flag_stars(source_fit.copy(),
                                             fwhm_lim=fwhm_lim,
                                             mag_lim=mag_lim)
                source_fit['guide_ind'] = guide_ind

                good_sources_df = source_fit[source_fit['bad_flag'] == False].reset_index(drop=True)

                if len(good_sources_df)>1:
                    return good_sources_df, guide_ind
                else:
                    continue
            else:
                continue
