#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 15:28:36 2021

@author: Briana
"""

import numpy as np
import numpy.ma as ma
import math
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt

# need to give it dataframe containing fiber information for one field with
# columns:
# flux, flux_err, ra, dec
# need to also supply fiber diameter, new pixel size, max distance, kernal
# sigma (degrees)
# optional to supply the new grid dimensions (xmin, xmax, ymin, ymax)
# otherwise builds grid from min and max coordinates of the field


class fibers_to_grid():

    def __init__(self, fib_ra, fib_dec, fiberd,
                 regrid_size, max_radius, kern_sig):
        self.fiberd = fiberd
        self.regrid_size = regrid_size
        self.max_dist = max_radius**2
        self.kern_sig = kern_sig

        self.fib_ra = fib_ra
        self.fib_dec = fib_dec

        self.fib_flux = None
        self.fib_flux_err = None

        self.grid_bounds = None
        self.new_pix_area = None
        self.regrid_x = None
        self.regrid_Y = None
        self.x_grid = None
        self.y_grid = None
        self.avg_dec = None

        self.nx = None
        self.ny = None

        self.weight_grid = None
        self.flux_grid = None
        self.error_grid = None

    def build_new_grid(self, grid=(0, 0, 0, 0)):

        try:
            xmin, xmax, ymin, ymax = grid
            if len(set(list(grid))) == 1:
                # min and max x,y values in the dither
                print('BUILDING grid from fiber coodinates')
                xmin, xmax = (self.fib_ra.min(), self.fib_ra.max())
                ymin, ymax = (self.fib_dec.min(), self.fib_dec.max())
        except:
            print('grid must be in format (xmin, xmax, ymin, ymax)')

        # adjust min and mas to include outside of edges fiber radius
        xmin = xmin - self.fiberd / 2.0
        xmax = xmax + self.fiberd / 2.0
        ymin = ymin - self.fiberd / 2.0
        ymax = ymax + self.fiberd / 2.0

        self.grid_bounds = (xmin, xmax, ymin, ymax)

        # size of pixels in new grid
        # need to multiply cos(dec) term to ra so get about same
        # number of pixels in x and y
        avg_y = np.average([ymax, ymin])
        self.avg_dec = avg_y
        self.regrid_x = self.regrid_size/np.cos(np.deg2rad(avg_y))
        self.regrid_y = self.regrid_size

        # defind number of dimenstions
        # nx, ny are the new grid dimensions
        self.nx = int((xmax-xmin)/(self.regrid_x))+1
        self.ny = int((ymax-ymin)/(self.regrid_y))+1

        # new pixel area
        self.new_pix_area = self.regrid_x*self.regrid_y

        # build new grid
        x_range = xmin+(np.arange(self.nx)+0.5)*self.regrid_x
        y_range = ymin+(np.arange(self.ny)+0.5)*self.regrid_y

        self.x_grid, self.y_grid = np.meshgrid(x_range, y_range)

        return self.x_grid, self.y_grid

    def shepards_kernal(self, fib_flux, fib_flux_err):
        if self.x_grid is None:
            self.build_new_grid()

        self.fib_flux = fib_flux
        self.fib_flux_err = fib_flux_err

        fib_sa = self.fib_flux/(((self.fiberd/2.0)**2)*math.pi)
        fib_sa_err = self.fib_flux_err/(((self.fiberd/2.0)**2)*math.pi)

        nx, ny = self.nx, self.ny
        xmin, ymin = self.grid_bounds[0], self.grid_bounds[2]

        I = np.zeros((ny, nx))
        W = np.zeros((ny, nx))
        E = np.zeros((ny, nx))

        # iterate through each pixel in the mastergrid
        for iy, ix in np.ndindex(I.shape):

            # find the x and y value of that pixel based on the grid size
            x = xmin + (ix + 0.5) * self.regrid_x
            y = ymin + (iy + 0.5) * self.regrid_y

            # distance modulous
            dist2_allfibs = ((x - self.fib_ra)**2) + ((y - self.fib_dec)**2)
            # find indices of fibers within user defined
            # max distance from pixel
            fib_inds = np.where(dist2_allfibs < self.max_dist)

            # if there are fibers within max distance from pixel
            if len(fib_inds[0]) > 0:

                dist2 = dist2_allfibs[fib_inds]
                sa = fib_sa[fib_inds]
                sa_err = fib_sa_err[fib_inds]

                nan_mask = np.where(~np.isnan(sa))
                dist2 = dist2[nan_mask]
                sa = sa[nan_mask]
                sa_err = sa_err[nan_mask]

                weights = np.exp(-0.5*dist2/(self.kern_sig**2))

                I[iy][ix] = np.sum(self.new_pix_area * weights * sa)
                E[iy][ix] = np.sum((self.new_pix_area**2) * (weights**2) * (sa_err**2))
                W[iy][ix] = np.sum(weights)

            # if no fibers within max distance of pixel
            else:

                I[iy][ix] = np.NaN
                E[iy][ix] = np.NaN
                W[iy][ix] = np.NaN

        self.weight_grid = W
        self.flux_grid = I/W
        self.error_grid = np.sqrt(E)/W

        return self.flux_grid, self.error_grid

    def build_wcs(self, wave_dict=None):
        npix_dec, npix_ra = np.shape(self.flux_grid)

        ra_ref_pix = int(npix_ra/2)
        dec_ref_pix = int(npix_dec/2)

        wcs_dict = {'CUNIT1': 'deg', 'CUNIT2': 'deg',
                    'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN',
                    'CRPIX1': ra_ref_pix, 'CRPIX2': dec_ref_pix,
                    'CRVAL1': self.x_grid[dec_ref_pix, ra_ref_pix],
                    'CRVAL2': self.y_grid[dec_ref_pix, ra_ref_pix],
                    'CDELT1': (self.regrid_x*-1), 'CDELT2': self.regrid_y}

        if isinstance(wave_dict, dict):
            wcs_dict = {**wcs_dict, **wave_dict}
            print('Creating 3D WCS')
        else:
            print('Creating 2D WCS')

        new_wcs = WCS(wcs_dict)
        return new_wcs

    def plot_results(self, plot_error=False, plot_weights=False,
                     savepath=None):

        fig, ax = plt.subplots(1, 1)
        ax.pcolor(self.x_grid, self.y_grid, self.flux_grid)
        ax.scatter(self.fib_ra, self.fib_dec, color='black', s=1)
        for i in range(len(self.fib_flux)):
            circle1 = plt.Circle((self.fib_ra[i], self.fib_dec[i]),
                                 self.fiberd/2.0, ec='grey', fill=False)
            ax.add_artist(circle1)

        plt.show()
        return None

    def save_interp_image(self, hdr, outfile):
        hdu_f = fits.PrimaryHDU(self.flux_grid, header=hdr)
        hdu_f.writeto(outfile, overwrite=True)
        hdu_f.close()

        hdu_e = fits.PrimaryHDU(self.error_grid, header=hdr)
        hdu_e.writeto(outfile, overwrite=True)
        hdu_e.close()

    def test_flux_conservation(self):

        num_fib = len(self.fib_flux)
        flux_grid_mask = ma.masked_invalid(self.flux_grid)
        num_pix = flux_grid_mask.count()
        print('# fibers:', num_fib, '# pixels:', num_pix)

        fib_area = num_fib*(((self.fiberd/2.0)**2)*np.pi)
        pix_area = num_pix*(self.regrid_x*self.regrid_y)
        print('total pix_area/fib_area:', pix_area/fib_area)

        pre_interp = np.sum(self.fib_flux)/fib_area
        post_interp = np.nansum(self.flux_grid)/pix_area
        convs_ratio = post_interp/pre_interp

        print('Flux pre-interp:', pre_interp)
        print('Flux post-interp:', post_interp)
        print('Flux_ratio:', convs_ratio)
        return convs_ratio
