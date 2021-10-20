#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 15:28:36 2021

@author: Briana
"""

import pandas as pd
import numpy as np
import datetime as dt
from scipy import interpolate

from astropy import coordinates as coords
from astropy.io import fits

import guider_observations
import IFU_spectrum
import image_utils


class VP_frame():
    def __init__(self, filename, fits_ext, fits_err_ext=3, grating='VP1',
                 cen_file='VP_config/IFUcen_VP_new_27m.csv', guide_obs=None,
                 run_correct_airmass=False):

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

        # determine if frame is binned 2x1 in spectral dimension
        if len(self.wave) < 2048:
            self.spec_bin = True
        else:
            self.spec_bin = False

        # determine which grating was used for frame
        self.grating = grating
        vp_grating_df = pd.read_csv('VP_config/VP_gratings.csv', skiprows=2)
        vp_grat_lis = list(vp_grating_df['grating'].values)
        if self.grating not in vp_grat_lis:
            raise ValueError(str(self.grating) + ' is not a VP grating'\
                             ' VP gratings:' + str(vp_grat_lis))
        self.grat_res = vp_grating_df[vp_grating_df['grating']==self.grating]['resolution'].values[0]

        if isinstance(guide_obs, guider_observations.guider_obs):
            self.match_guider_frames(guide_obs)
        else:
            self.guider_ind = []
            self.guide_match = False

        # correct spec for airmass if it has not already been done
        # according to header keyword am_corr
        # and run_correct_airmass is True
        self.am_corr = self.hdr['AMCORR']
        if not self.am_corr:
            if run_correct_airmass:
                self.correct_airmass()
                self.corr_am = True

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
      
            if 'AM_CORR' not in self.hdr:
                self.hdr['AMCORR'] = (False, 'True if corrected for airmass')
            else:
                self.hdr['AMCORR'] = self.hdr['AM_CORR']

            self.dat_err = hdulis[self.fits_err_ext].data

            self.num_fibs = self.hdr['NAXIS2']
            self.num_wl = self.hdr['NAXIS1']
            self.exptime = self.hdr['EXPTIME']
            self.airmass = self.hdr['AIRMASS']

            self.RA, self.DEC = image_utils.coord_hms_to_deg(self.hdr['RA'],
                                                             self.hdr['DEC'])

            self.wave_start = self.hdr['CRVAL1']
            self.wave_delta = self.hdr['CDELT1']
            self.wave_end = self.hdr['CRVAL1'] + ((self.hdr['NAXIS1']-1) * self.hdr['CDELT1'])

            self.wave = self.hdr['CRVAL1'] + ((np.arange(self.hdr['NAXIS1']) * self.hdr['CDELT1']))

            self.fib_df = self.fib_df[0:self.num_fibs]
            self.fib_df['dith_num'] = self.dith_num

    def match_guider_frames(self, guide_obs):

        if isinstance(guide_obs, guider_observations.guider_obs):

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
            raise ValueError('guide_obs must but be a guider_observations class object '+str(type(guide_obs)))

    # new_ext_name (int/str): name of new fits extension
    # hdr_comment (str): comment to add to header of new extension
    def build_new_extension(self, new_ext_name, hdr_comment, err=False):

        hdulis = fits.open(self.filename, lazy_load_hdus=False)
        
        if err:
            dat_new = self.dat_err.copy()
            new_ext_name = new_ext_name+'_err'
        else:
            dat_new = self.dat.copy()
        hdr_new = self.hdr.copy()
        hdr_new['EXTNAME'] = new_ext_name
        hdr_new['comment'] = hdr_comment
        new_hdu = fits.ImageHDU(dat_new, header=hdr_new)

        # first check if extension exists
        # if it does overwrite extension with new dat+hdr
        try:
            fits.update(self.filename, new_hdu.data, new_hdu.header, new_ext_name, output_verify='silentfix')
            print('OVERWRITING fits extension: [' + str(self.fits_name) +'][EXT:' + str(new_ext_name) + ']' )

        # else if extension does not exist create new extension
        except KeyError:
            hdulis.append(new_hdu)
            hdulis.writeto(self.filename, overwrite=True, checksum=True,
                           output_verify='silentfix')
            print('BUILDING new fits extension: [' + str(self.fits_name) +'][EXT:' + str(new_ext_name) + ']' )

        hdulis.close()

    def correct_airmass(self, obs_ext='VP_config/McD_extinction_curve.csv',
                        atm=1.0):

        try:
            mcd_ext = pd.read_csv(obs_ext)
            ext_wave = mcd_ext['wave']
            ext = mcd_ext['ext(mag/AM)']*atm

            min_ew, max_ew = np.amin(ext_wave), np.amax(ext_wave)

            if (self.wave_start > min_ew) & (self.wave_end < max_ew):
                f_ext = interpolate.interp1d(x=ext_wave, y=ext)
                ext_coeff = f_ext(self.wave)

                dat_am_corr = np.zeros(np.shape(self.dat))
                dat_err_am_corr = np.zeros(np.shape(self.dat_err))
                # extinct spec for airmass
                for i in range(len(self.dat)):
                    dat_am_corr[i] = (self.dat[i])*(10.0**(-0.4*ext_coeff*self.airmass))
                    dat_err_am_corr[i] = (self.dat_err[i])*(10.0**(-0.4*ext_coeff*self.airmass))

                self.dat = dat_am_corr
                self.dat_err = dat_err_am_corr
                self.hdr['AMCORR'] = True
                self.am_corr = True

                new_ext_name = 'am_corr'
                ext_hdr_com = 'airmass corrected spec'
                self.build_new_extension(new_ext_name, ext_hdr_com)
                self.build_new_extension(new_ext_name, ext_hdr_com, err=True)

                self.fits_ext = 'am_corr'
                self.fits_err_ext = 'am_corr_err'

            else:
                raise ValueError('Exctinction curve must encompass data bounds')

        except FileNotFoundError:
            print('could not find: '+str(obs_ext))

        except KeyError:
            print('need to have columns: wave, ext(mag/AM)')

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
            raise ValueError('Must provide sky model with shape:'
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

        sum_spec = IFU_spectrum.spectrum(spec, self.wave, z=z)

        if plot:
            sum_spec.plot_spec(spec_units='Electrons per second')

        return sum_spec
