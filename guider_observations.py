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
import warnings

import image_utils

from astropy.wcs import WCS
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.utils.exceptions import AstropyWarning


class guider_obs():
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

    def open_guider_frame(self, guider_name, fix_hdr_wcs=True):
        if isinstance(guider_name, (np.int64, int, float)):
            if guider_name < len(self.guider_df):
                hdu = fits.open(self.guider_df.iloc[abs(guider_name)]['filename'])
            else:
                raise ValueError('Guider index out of bounds')

        elif isinstance(guider_name, str):
            try:
                hdu = fits.open(guider_name)
            except ValueError:
                raise ValueError('Invalid guider filename, must be fits file or guider index integer')

        else:
            raise ValueError('Need to provide guider path/filename.fits (str) or index (int) of guider frame in guider_df\
                             provided: '+str(guider_name)+' of type: '+str(type(guider_name)))

        dat = hdu[0].data
        hdr = hdu[0].header
        hdu.close()

        if fix_hdr_wcs:
            # define list of header keys incorrectly defined as str in headers
            # these need to be corrected to floats to be read by WCS
            wcs_hdr_keys = ['CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2']
            for w in wcs_hdr_keys:
                hdr[w] = float(hdr[w])
            hdr['CDELT1'] = -1*self.guider_ps/3600  #convert to deg
            hdr['CDELT2'] = self.guider_ps/3600  #convert to deg
            del hdr['CD1_1']
            del hdr['CD1_2']
            del hdr['CD2_1']
            del hdr['CD2_2']
            

        return dat, hdr

    # guider_name (int/str): can be either guider index in
    # guider_df or filename
    def inspect_guider_frame(self, guider_name, vmin=None, vmax=None):
        dat, hdr = self.open_guider_frame(guider_name)
        im_wcs = WCS(hdr)
        image_utils.plot_frame(dat, im_wcs, vmin=vmin, vmax=vmax)

    def find_guide_stars(self, guider_name, star_thres=10.,
                         num_bright_stars=10, star_fwhm=8.0,
                         plot_guide_frame=False):

        g_dat, g_hdr = self.open_guider_frame(guider_name)

        sources_df = image_utils.find_stars(g_dat, star_thres=star_thres,
                                            num_bright_stars=num_bright_stars,
                                            star_fwhm=star_fwhm,
                                            plot_sources=plot_guide_frame)

        if sources_df is None:
            return None
        else:
            return sources_df.copy()

    def get_guide_stars_mag(self, guider_name):
        dat, hdr = self.open_guider_frame(guider_name)
        im_wcs = WCS(hdr)

        cat_df = image_utils.get_star_mag(dat, im_wcs)
        return cat_df

    def measure_guide_star_params(self, guider_name, sources_df,
                                  plot_star_cutouts=False):

        g_dat, g_hdr = self.open_guider_frame(guider_name)
        # divide guider image by exptime to get counts per second
        guide_et = float(g_hdr['EXPTIME'])
        g_dat_perS = g_dat/guide_et  # counts per second

        # remove background (approx bias) level from guider image
        mean, median, std = sigma_clipped_stats(g_dat_perS, sigma=3.0)
        g_dat_perS_noBKGD = g_dat_perS - median

        sources_df = image_utils.measure_star_params(g_dat_perS_noBKGD, sources_df,
                                                     plot_star_cutouts=plot_star_cutouts)

        sources_df['fwhm(arcseconds)'] = sources_df['fwhm(pixels)']*self.guider_ps

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
            guide_ind = int(guide_ind_list[g])

            source_ex = self.find_guide_stars(guide_ind, star_thres=star_thres,
                                              num_bright_stars=num_bright_stars,
                                              star_fwhm=star_fwhm)
            if source_ex is not None:
                source_fit = self.measure_guide_star_params(guide_ind,
                                                            source_ex)

                source_fit = self.flag_stars(source_fit.copy(), fwhm_lim=fwhm_lim, mag_lim=mag_lim)

                source_fit['guide_ind'] = guide_ind

                good_sources_df = source_fit[source_fit['bad_flag'] == False].reset_index(drop=True)

                if len(good_sources_df) > 1:
                    return good_sources_df, guide_ind
                else:
                    continue
            else:
                continue
            
            print('No reference guide frame found')
            return None, None
