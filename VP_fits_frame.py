#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 15:28:36 2021

@author: Briana
"""

import pandas as pd
import numpy as np
import datetime as dt

from astropy import coordinates as coords
from astropy.io import fits

import guider_observations as go
import IFU_spectrum as ifu_spec


class VP_fits_frame():
    def __init__(self, filename, fits_ext, fits_err_ext=3,
                 cen_file='VP_config/IFUcen_VP_new_27m.csv', guide_obs=None):

        self.filename = filename
        self.fits_ext = fits_ext
        self.fits_err_ext = fits_err_ext
        self.fib_df = pd.read_csv(cen_file, skiprows=2)

        self.fits_name = self.filename.split('/')[-1]
        self.obs_datetime = dt.datetime.strptime(self.fits_name.split('_')[-2],
                                                 '%Y%m%dT%H%M%S')

        if 'dither' in self.fits_name.split('_'):
            self.dith_num = int(self.fits_name.split('_')[-3])
            self.object = '_'.join(self.fits_name.split('_')[0:-4])
        else:
            self.dith_num = None
            self.object = '_'.join(self.fits_name.split('_')[0:-2])

        self.seeing = 999
        self.dithnorm = 1.0

        self.build_hdr_info()

        if isinstance(guide_obs, go.guider_observations):
            self.match_guider_frames(guide_obs)
        else:
            self.guider_ind = []
            self.guide_match = False

        self.dither_group_id = None

        print('BUILD VP science frame [' + str(self.fits_name) +
              '][EXT:' + str(self.fits_ext) + ']')

    def build_hdr_info(self):
        with fits.open(self.filename) as hdulis:
            try:
                self.dat = hdulis[self.fits_ext].data
                self.dat_err = hdulis[self.fits_err_ext].data
            except KeyError:
                print(self.fits_ext, 'does not exist')

            self.hdr = hdulis[self.fits_ext].header

            if 'SEEING' not in self.hdr:
                self.hdr['SEEING'] = (self.seeing, '(arcseconds) measured \
                        from guide stars')
                self.hdr['DITHNORM'] = (self.dithnorm, 'dither normalization \
                        from guide stars')
            else:
                self.seeing = self.hdr['SEEING']
                self.dithnorm = self.hdr['DITHNORM']

            self.dat_err = hdulis[self.fits_err_ext].data

            self.num_fibs = self.hdr['NAXIS2']
            self.num_wl = self.hdr['NAXIS1']
            self.exptime = self.hdr['EXPTIME']
            self.airmass = self.hdr['AIRMASS']

            ra = self.hdr['RA'].split(':')
            dec = self.hdr['DEC'].split(':')
            ra_str = ra[0]+'h'+ra[1]+'m'+ra[2]+'s'
            dec_str = dec[0]+'d'+dec[1]+'m'+dec[2]+'s'
            im_coords = coords.SkyCoord(ra_str, dec_str, frame='icrs')

            self.RA = im_coords.ra.deg
            self.DEC = im_coords.dec.deg

            self.wave_start = self.hdr['CRVAL1']
            self.wave_delta = self.hdr['CDELT1']
            self.wave_end = self.hdr['CRVAL1'] + ((self.hdr['NAXIS1']-1) * self.hdr['CDELT1'])

            self.wave = self.hdr['CRVAL1'] + ((np.arange(self.hdr['NAXIS1']) * self.hdr['CDELT1']))

            self.fib_df = self.fib_df[0:self.num_fibs]
            self.fib_df['dith_num'] = self.dith_num

    def match_guider_frames(self, guide_obs):
        
        if isinstance(guide_obs, go.guider_observations):
            
            guider_df = guide_obs.guider_df

            d_dt = self.obs_datetime
            d_et = dt.timedelta(0, self.exptime)
            # find guider frames taken within same window as science exposure
            # store the guider_df indices for matching guider frames to data_df
            gmatch = guider_df[(guider_df['obs_datetime']>=d_dt) & (guider_df['obs_datetime']<=(d_dt+d_et))]
            gmatch_inds = gmatch.index.values
            self.guider_ind = gmatch_inds
            self.guide_match = True
            if len(gmatch_inds) < 1:
                print('WARNING: No guider frames found for', self.filename)

        else:
            print('guide_obs but be a guider_observations class object')

    # new_ext_name (int/str): name of new fits extension
    # hdr_comment (str): comment to add to header of new extension
    def build_new_extension(self, new_ext_name, hdr_comment):

        hdulis = fits.open(self.filename, lazy_load_hdus=False)
        
        dat_new = self.dat.copy()
        hdr_new = self.hdr.copy()
        hdr_new['EXTNAME'] = new_ext_name
        hdr_new['comment'] = hdr_comment
        new_hdu = fits.ImageHDU(dat_new, header=hdr_new)

        # first check if extension exists
        # if it does overwrite extension with new dat+hdr
        try:
            fits.update(self.filename, new_hdu.data, new_hdu.header, 'dithnorm', output_verify='silentfix')
            print('OVERWRITING fits extension: [' + str(self.fits_name) +'][EXT:' + str(new_ext_name) + ']' )

        # else if extension does not exist create new extension
        except KeyError:
            hdulis.append(new_hdu)
            hdulis.writeto(self.filename, overwrite=True, checksum=True,
                           output_verify='silentfix')
            print('BUILDING new fits extension: [' + str(self.fits_name) +'][EXT:' + str(new_ext_name) + ']' )

        hdulis.close()

    # sky_model (array): user provided sky model
    # with same shape and wave sol as data spectra
    def new_sky_subtract(self, sky_model, new_ext_name=None, hdr_comment=None):
        if np.shape(sky_model)[0] == np.shape(self.dat)[1]:

            sky_sub_dat = np.zeros(np.shape(self.dat))
            for i in range(self.num_fibs):
                sky_sub_dat = self.dat[i] - sky_model

            self.dat = sky_sub_dat

            if new_ext_name is None:
                new_ext_name = 'user_sky_sub'
            if hdr_comment is None:
                hdr_comment = 'sky sub with user provided sky model'
            self.build_new_extension(new_ext_name, hdr_comment)

        else:
            print('Must provide sky model with shape:'
                  + str(np.shape(self.dat)[0]))

    # fib_ind (list/array)(optional, default will sum all fibers):
    # list of fiber indices to sum and plot
    def build_frame_sum_spec(self, fib_inds=[], z=np.NaN, plot=False):
        if len(fib_inds) == 0:
            spec = np.sum(self.dat, axis=0)
        elif len(fib_inds) == 1:
            spec = self.dat[fib_inds[0]]
        else:
            spec = np.sum(self.dat[fib_inds, :], axis=0)

        sum_spec = ifu_spec.spectrum(spec, self.wave, z=z)

        if plot:
            sum_spec.plot_spec(spec_units='Electrons per second')

        return sum_spec
