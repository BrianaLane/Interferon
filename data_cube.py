#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 11:07:41 2021

@author: Briana
"""

import pandas as pd
import numpy as np

# from astropy import units as u
from astropy.io import fits
# from astropy.coordinates import Angle


class data_cube():

    def __init__(self, cube_file, err_cube_file=None):
        hdu = fits.open(cube_file)
        self.hdr = hdu[0].header
        self.dat = hdu[0].data
        hdu.close()

        self.grid = None
        self.wave = None

        if err_cube_file is not None:
            hdu_err = fits.open(err_cube_file)
            self.dat_err = hdu_err[0].data
            hdu_err.close()
        else:
            self.dat_err = None
