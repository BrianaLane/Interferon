#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 15:28:36 2021

@author: Briana
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


class spectrum():

    def __init__(self, spec, wave, z=None, obj_name=None):
        self.spec = spec
        self.wave = wave

        self.obj_name = obj_name
        self.z = z

    def plot_spec(self, spec_units='Flux'):
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))

        if ~(np.isnan(self.z)):
            ax.plot(self.wave*(1+self.z), self.spec, lw=2, color='black')
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

    def fit_lines(self):
        return None
