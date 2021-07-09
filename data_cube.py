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


class data_cube():

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

    def collapse_frame(self, wave_range=5007):
        if isinstance(wave_range, int) or isinstance(wave_range, float):
            wave_inds = np.where(self.wave == wave_range)
        elif isinstance(wave_range, tuple):
            wave_inds = np.where((self.wave > np.min(wave_range)) &
                                 (self.wave < np.max(wave_range)))
        col_frame = self.cube[wave_inds]
        col_err_frame = self.cube_err[wave_inds]

        return col_frame, col_err_frame
    
    def plot_frame(self, col_frame, name='', save=False):
        fig, ax = plt.subplots(1, 1, figsize=(10,10))
        ax.imshow(col_frame, origin='lower')
        ax.set_xlabel('Right Ascension (J'+str(self.equinox)+')')
        ax.set_ylabel('Declination (J'+str(self.equinox)+')')
        
        if save:
            plt.savefig(self.filename.split('.fits')[0]+'_collapse_im_'+name+'_.png')
            
        plt.show()
        
    def save_frame(self, col_frame):
        hdu_new = fits.PrimaryHDU(fits_cube, header=wcs_hdr)
