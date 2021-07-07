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
from astropy import units as u
from astropy.coordinates import Angle

import guider_observations as go
import IFU_spectrum as ifu_spec
import interpolate_IFU


class VP_fits_frame():
    def __init__(self, filename, fits_ext, fits_err_ext=3,
                 cen_file='VP_config/IFUcen_VP_new_27m.csv', guide_obs=None):

        self.filename = filename
        self.fits_ext = fits_ext
        self.fits_err_ext = fits_err_ext
        self.fib_df = pd.read_csv(cen_file, skiprows=2)

        fits_name = self.filename.split('/')[-1]
        self.obs_datetime = dt.datetime.strptime(fits_name.split('_')[-2],
                                                 '%Y%m%dT%H%M%S')

        if fits_name.split('_')[-4] == 'dither':
            self.dith_num = int(fits_name.split('_')[-3])
            self.object = '_'.join(fits_name.split('_')[0:-4])
        else:
            self.dith_num = None
            self.object = '_'.join(fits_name.split('_')[0:-2])

        self.seeing = 999
        self.dithnorm = 1.0

        self.build_hdr_info()

        if isinstance(guide_obs, go.guider_observations):
            self.match_guider_frames(guide_obs)
        else:
            self.guider_ind = None
            self.guide_match = None

        self.dither_group_id = None

        self.frame_cube = None
        self.frame_err_cube = None

    def build_hdr_info(self):
        with fits.open(self.filename) as hdulis:
            self.dat = hdulis[self.fits_ext].data
            self.dat_err = hdulis[self.fits_err_ext].data
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
            if len(gmatch_inds) < 1:
                self.guide_match = False
                print('WARNING: No guider frames found for', self.filename)
            else:
                self.guide_match = True

            self.match_guide_frames = True
        else:
            print('guide_obs but be a guider_observations class object')

    # new_ext_name (int/str): name of new fits extension
    # hdr_comment (str): comment to add to header of new extension
    def build_new_extension(self, new_ext_name, hdr_comment):

        hdulis = fits.open(self.filename, lazy_load_hdus=False)

        # first check if extension exists
        # if it does overwrite extension with new dat+hdr
        try:
            hdulis[new_ext_name].data = self.dat
            hdulis[new_ext_name].data = self.hdr

            hdulis.writeto(self.filename, overwrite=True)
            print('Overwriting:', new_ext_name, 'extension')

        # else if extension does not exist create new extension
        except:
            hdu_new = fits.ImageHDU(self.dat, header=self.hdr)
            hdulis_new = fits.HDUList(hdulis+[hdu_new])

            hdulis_new[-1].header['EXTNAME'] = new_ext_name
            hdulis_new[-1].header['comment'] = hdr_comment

            hdulis_new.writeto(self.filename, overwrite=True)
            hdulis_new.close()

        hdulis.close()

        # rebuild hdr information with new extension details
        self.build_hdr_info()

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
     
    def build_data_cube(self, grid=(0, 0, 0, 0)):

        fiberd_as = self.cen_df.iloc[0]['fiberd']  # in arcseconds
        fibersep_as = self.cen_df.iloc[0]['fibersep']  # in arcseconds
        # convert all arcsecond units to degrees
        fiberd = Angle(fiberd_as*u.arcsecond).degree
        fibersep = Angle(fibersep_as*u.arcsecond).degree

        fib_RA = self.master_fib_df['RA'].values
        fib_DEC = self.master_fib_df['DEC'].values

        regrid_size = (fiberd+fibersep)/2.0
        kern_sig = fiberd+fibersep
        max_radius = (fiberd+fibersep)*5.0
        interp_class = interpolate_IFU.fibers_to_grid(fib_RA, fib_DEC, fiberd,
                                                      regrid_size, max_radius,
                                                      kern_sig)

        if len(set(list(grid))) == 1:
            xmin = self.fib_df['RA'].min()
            xmax = self.fib_df['RA'].max()
            ymin = self.fib_df['DEC'].min()
            ymax = self.fib_df['DEC'].max()
            grid = (xmin, xmax, ymin, ymax)

        x_grid, y_grid = interp_class.build_new_grid(grid=grid)

        self.master_fib_df['fib_flux'] = np.ones(len(self.fib_df))
        self.master_fib_df['fib_flux_err'] = np.zeros(len(self.fib_df))

        wave_frame_lis = []
        wave_frame_err_lis = []
        for i in range(len(self.wave)):
            self.fib_df['fib_flux'] = self.dat[:, i]
            self.fib_df['fib_flux_err'] = self.dat_err[:, i]

            wave_frame, wave_err_frame = interp_class.shepards_kernal()
            wave_frame_lis.append(wave_frame)
            wave_frame_err_lis.append(wave_err_frame)

        self.frame_cube = np.dstack(wave_frame_lis)
        self.frame_err_cube = np.dstack(wave_frame_err_lis)

    def save_data_cube(self, outname=None):

        hdu_new = fits.PrimaryHDU(self.data_cube)
        hdr_new = hdu_new[0].header
        hdr_new['OBJECT'] = self.hdr['OBJECT']

        if outname is None:
            outname = self.filename[0:-11]+'data_cube.fits'

        hdu_new.writeto('outname')
        hdu_new.close()

    # fib_ind (list/array)(optional, default will sum all fibers):
    # list of fiber indices to sum and plot
    def build_frame_sum_spec(self, fib_inds=[], z=np.NaN, plot=False):
        if len(fib_inds) == 0:
            spec = np.sum(self.dat, axis=0)
        elif len(fib_inds) == 1:
            spec = self.dat[fib_inds[0]]
        else:
            spec = np.sum(self.dat[fib_inds, :], axis=0)

        sum_spec = ifu_spec.IFU_spectrum(spec, self.wave, z=z)

        if plot:
            sum_spec.plot_spec(spec_units='Electrons per second')

        return sum_spec
