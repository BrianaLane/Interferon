import os.path as op
import pandas as pd
import numpy as np
import math
import glob
import datetime as dt
import matplotlib.pyplot as plt
import warnings
import difflib as dl
import warnings
import sys

from scipy import interpolate

import astropy
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

import interpolate_IFU


class IFU_spectrum():
    def __init__(self, spec, wave, z=None, obj_name=None):
        self.spec = spec
        self.wave = wave
        
        self.obj_name = obj_name
        self.z = z 
        
    def plot_spec(self, spec_units='Flux'):            
        fig, ax = plt.subplots(1,1, figsize=(15,5))
        
        if ~(np.isnan(self.z)):
            ax.plot(self.wave*(1+z), self.spec, lw=2, color='black')
            ax.set_xlabel(r'Rest Wavelength ($\AA$)', fontsize=25)
            
        else:
            ax.plot(self.wave, self.spec, lw=2, color='black')
            ax.set_xlabel(r'Observed Wavelength ($\AA$)', fontsize=20)
            
        ax.set_ylabel(spec_units, fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=15)
        
        plt.show()
        
    def new_wave_sol(self, new_wave):
        f = interpolate.interp1d(self.wave, self.spec)
        new_spec = f(new_wave)
        self.spec = new_spec
        
    def fit_emission_lines(self):
        return None