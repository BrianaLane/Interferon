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

'''
Shepards Kernal Interpolation class
'''


#need to give it dataframe containing fiber information for one field with columns:
#  flux, flux_err, ra, dec
#need to also supply fiber diameter, new pixel size, max distance, kernal sigma (degrees)
#optional to supply the new grid dimensions (xmin, xmax, ymin, ymax)
#   otherwise builds grid from min and max coordinates of the field

class fibers_to_grid():
	def __init__(self, fib_flux, fib_flux_err, fib_ra, fib_dec, fiberd, regrid_size, max_radius, kern_sig):
		self.fiberd      = fiberd
		self.regrid_size = regrid_size
		self.max_dist    = max_radius**2
		self.kern_sig    = kern_sig

		self.fib_ra       = fib_ra
		self.fib_dec      = fib_dec

		self.fib_flux     = fib_flux
		self.fib_flux_err = fib_flux_err
		self.fib_sa       = self.fib_flux/(((self.fiberd/2.0)**2)*math.pi) 
		self.fib_sa_err   = self.fib_flux_err/(((self.fiberd/2.0)**2)*math.pi) 

		self.grid_bounds = None
		self.new_pix_area = None
		self.regrid_x = None
		self.regrid_Y = None
		self.x_grid = None
		self.y_grid = None

		self.nx = None
		self.ny = None
		self.avg_dec = None

		self.weight_grid = None
		self.flux_grid = None
		self.error_grid = None

	def build_new_grid(self, grid=None):
		if grid == None:
			#min and max x,y values in the dither 
			xmin, xmax = (self.fib_ra.min(), self.fib_ra.max())
			ymin, ymax = (self.fib_dec.min(), self.fib_dec.max())
		else:
			try:
				xmin, xmax, ymin, ymax = grid 
			except:
				print('grid must be in format (xmin, xmax, ymin, ymax)')

		#adjust min and mas to include outside of edges fiber radius
		xmin = xmin - self.fiberd/2.0
		xmax = xmax + self.fiberd/2.0
		ymin = ymin - self.fiberd/2.0
		ymax = ymax + self.fiberd/2.0

		self.grid_bounds = (xmin, xmax, ymin, ymax)

		#size of pixels in new grid
		#need to add cos term to dec so get about same # pixels in x and y
		avg_y = np.average([ymax, ymin])
		self.avg_dec = avg_y
		self.regrid_x = self.regrid_size/np.cos(np.deg2rad(avg_y))
		self.regrid_y = self.regrid_size

		#defind number of dimenstions 
		#nx, ny are the new grid dimensions
		self.nx = int((xmax-xmin)/(self.regrid_x))+1
		self.ny = int((ymax-ymin)/(self.regrid_y))+1

		#new pixel area
		self.new_pix_area = self.regrid_x*self.regrid_y

		#build new grid 
		x_range = xmin+(np.arange(self.nx)+0.5)*self.regrid_x
		y_range = ymin+(np.arange(self.ny)+0.5)*self.regrid_y
		self.x_grid, self.y_grid = np.meshgrid(x_range, y_range)

		return self.x_grid, self.y_grid

	def shepards_kernal(self):
		nx, ny = self.nx, self.ny
		xmin, ymin = self.grid_bounds[0], self.grid_bounds[2]

		I = np.zeros((ny,nx))
		W = np.zeros((ny,nx))
		E = np.zeros((ny,nx))

		#iterate through each pixel in the mastergrid
		for iy,ix in np.ndindex(I.shape):

			#find the x and y value of that pixel based on the grid size 
			x=xmin+(ix+0.5)*self.regrid_x
			y=ymin+(iy+0.5)*self.regrid_y

			#distance modulous
			dist2_allfibs = ((x - self.fib_ra)**2) + ((y - self.fib_dec)**2)
			#find indices of fibers within user defined max distance from pixel
			fib_inds = np.where(dist2_allfibs < self.max_dist)

			#if there are fibers within max distance from pixel
			if len(fib_inds[0]) > 0:

				dist2 = dist2_allfibs[fib_inds]
				sa = self.fib_sa[fib_inds]
				sa_err = self.fib_sa_err[fib_inds]

				weights = np.exp(-0.5*dist2/(self.kern_sig**2))

				I[iy][ix] = np.sum(self.new_pix_area * weights * sa)
				E[iy][ix] = np.sum((self.new_pix_area**2) * (weights**2) * (sa_err**2))
				W[iy][ix] = np.sum(weights)

			#if no fibers within max distance of pixel
			else:

				I[iy][ix] = np.NaN
				E[iy][ix] = np.NaN
				W[iy][ix] = np.NaN

		self.flux_grid = I/W
		self.error_grid = np.sqrt(E)/W

		return self.flux_grid, self.error_grid

		def plot_results(plot_error=False, plot_weights=False, savepath=None):
			fig, ax = plt.subplots(1,1)
			ax.pcolor(self.x_grid, self.y_grid, self.flux_grid)
			ax.scatter(self.fib_ra, self.fib_dec, color='black', s=1)
			for i in range(len(self.fib_flux)):
				circle1 = plt.Circle((self.fib_ra[i], self.fib_dec[i]), self.fiberd/2.0, ec='grey', fill=False)
				ax.add_artist(circle1)
			#ax.set_ylim(1-max_radius-1, size+max_radius+1)
			#ax.set_xlim(1-max_radius-1, size+max_radius+1)
			plt.show()
			return None

		def save_interp_image(self, outpath, filename):
			hdu_f = fits.PrimaryHDU(self.flux_grid)
			hdu_f.writeto(op.join(outpath, filename+'.fits'), overwrite=True)

			hdu_e = fits.PrimaryHDU(self.error_grid)
			hdu_e.writeto(op.join(outpath, filename+'_e.fits'), overwrite=True)

		def test_flux_conservation(self):

			num_fib = len(self.fib_flux)
			flux_grid_mask  = ma.masked_invalid(self.flux_grid)
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


