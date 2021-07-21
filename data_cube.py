#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 11:07:41 2021

@author: Briana
"""
import numpy as np

# from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
# from astropy.coordinates import Angle

import image_utils
from IFU_spectrum import spectrum


class Cube():

    def __init__(self, cube_file, err_cube_file=None):

        hdu = fits.open(cube_file)
        self.filename = cube_file
        self.hdr = hdu[0].header
        self.cube = hdu[0].data
        hdu.close()

        self.object = self.hdr['OBJECT']
        self.RA = self.hdr['RA']
        self.DEC = self.hdr['DEC']
        self.equinox = self.hdr['EQUINOX']

        self.wcs = WCS(self.hdr)
        self.grid = None
        self.wave = self.hdr['CRVAL3'] + ((np.arange(self.hdr['NAXIS3']) * self.hdr['CDELT3']))

        if err_cube_file is not None:
            hdu_err = fits.open(err_cube_file)
            self.cube_err = hdu_err[0].data
            hdu_err.close()
        else:
            self.cube_err = None

    def collapse_frame(self, wave_range=None, err=False, save=False):

        if isinstance(wave_range, int) or isinstance(wave_range, float):
            wave_inds = np.where(self.wave == wave_range)
            wavename = str(wave_range)

        elif isinstance(wave_range, tuple):
            wave_inds = np.where((self.wave > np.min(wave_range)) &
                                 (self.wave < np.max(wave_range)))
            wavename = '_range_'+str(np.min(wave_range))+'_'+str(np.max(wave_range))

        else:
            #if no wave range provide collapse entire cube into 2d frame
            wave_inds = np.arange(len(self.wave))
            wave_name = 'all'

        if not err:
            col_frame = np.sum(self.cube[wave_inds], axis=0)

        else:
            if isinstance(self.cube_err, np.array):
                col_frame = np.sum(self.cube_err[wave_inds], axis=0)
            else:
                print('NO ERROR CUBE PROVIDED')
                return None
            
        if save:
            #add giving it the cube header but only 2D WCS
            hdu = fits.PrimaryHDU(col_frame)            
            outname = self.filename.split('.fits')[0]+'_collpasecube_'+wave_name+'.fits'
            print('SAVING collpased cube: '+outname)
            hdu.writeto(outname, overwrite=True)
                
        return col_frame

    def collapse_spectrum(self, err=False):

        if not err:
            sum_spec = np.sum(self.cube, axis=0)

        else:
            if isinstance(self.cube_err, np.array):
                sum_spec = np.sum(self.cube_err, axis=0)
            else:
                print('NO ERROR CUBE PROVIDED')
                return None

        spec_obj = spectrum(sum_spec, self.wave, z=None, obj_name=self.object)

        return spec_obj

    def extract_spectrum(self, RA, DEC, apert_rad, err=False):

        if not err:
            sum_spec = np.sum(self.cube, axis=0)

        else:
            if isinstance(self.cube_err, np.array):
                sum_spec = np.sum(self.cube_err, axis=0)
            else:
                print('NO ERROR CUBE PROVIDED')
                return None

        spec_obj = spectrum(sum_spec, self.wave, z=None, obj_name=self.object)

        return spec_obj

    def build_sensitiviy_curve(self):
        # find the center of mass of star
        # call extract spectrum to build spectrum of star
        # could decide on ap_rad by fitting profile to image slice
        # find lit spectrum to compare
        # divide to get sensitivty curve
        # correct for airmass
        return None

    def flux_calibrate(self):
        return None

    def regrid_cube(self, apert_rad):
        # may not make sense here b/c would need master fiber data
        # could write this into the object
        return None
