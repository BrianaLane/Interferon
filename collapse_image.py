#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 16:13:46 2021

@author: Briana
"""

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS


class cube_image():

    def __init__(self, im_file):

        hdu = fits.open(im_file)
        self.filename = im_file
        self.hdr = hdu[0].header
        self.cube = hdu[0].data
        hdu.close()

        self.object = self.hdr['OBJECT']
        self.RA = self.hdr['RA']
        self.DEC = self.hdr['DEC']
        self.equinox = self.hdr['EQUINOX']

        self.wcs = WCS(self.hdr)
        self.grid = None
        self.wave_range = None

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