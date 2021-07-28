#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 11:07:41 2021

@author: Briana
"""
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS, utils
from astropy import units as u
from astropy.coordinates import SkyCoord

import image_utils as imu
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
        self.wcs_2d = WCS(self.hdr, naxis=[1, 2])
        self.x_grid, self.y_grid = np.meshgrid(np.arange(np.shape(self.cube)[2]),
                                               np.arange(np.shape(self.cube)[1]))
        self.coord_grid = utils.pixel_to_skycoord(self.x_grid, self.y_grid,
                                                  self.wcs_2d)
        self.pix_scale = np.NaN 
        self.wave = self.hdr['CRVAL3'] + ((np.arange(self.hdr['NAXIS3']) * self.hdr['CDELT3']))

        if err_cube_file is not None:
            try:
                hdu_err = fits.open(err_cube_file)
                self.cube_err = hdu_err[0].data
                hdu_err.close()
            except:
                print('INVALID ERROR CUBE FILE PROVIDED')
                self.cube_err = None
        else:
            self.cube_err = None
            
        self.col_cube_im = None
        self.col_cube_spec = None

    def build_2D_hdr():
        # relies on the assumption the header for the data cube
        # is formatted so that the wcs information is written right
        # below the basic header keywords so it looks for the end of the
        # WCS keywords by looking for 'RADESYS'='ICRS' and copies everything below
        cube_hdr_vals = [self.hdr[i] for i in range(len(self.hdr))]
        extra_ind = cube_hdr_vals.index('ICRS')
        cube_hdr_extra = self.hdr[extra_ind+1::]
    
        wcs_2d_hdr = self.wcs_2d.to_header()
    
        hdu_new = fits.PrimaryHDU(self.cube[0, :, :])
        im_hdr = hdu_new.header+wcs_2d_hdr+cube_hdr_extra
        im_hdr['COMMENT'] = 'BUILD 2D image from data cube'
    
        return im_hdr

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
            if self.cube_err is not None:
                col_frame = np.sum(self.cube_err[wave_inds], axis=0)
            else:
                print('ERROR: no error cube provided')
                return None
            
        if (wave_name == 'all') and (err==False):
            self.col_cube_im = col_frame
            
        if save:
            #add giving it the cube header but only 2D WCS
            hdr_im = self.build_2D_hdr()
            hdu = fits.PrimaryHDU(col_frame, header=hdr_im)            
            outname = self.filename.split('.fits')[0]+'_collpasecube_'+wave_name+'.fits'
            if err:
                outname = outname[0:-5]+'_err.fits'
            print('SAVING collpased cube: '+outname)
            hdu.writeto(outname, overwrite=True)
                
        return col_frame

    def collapse_spectrum(self, err=False, save=False):

        if not err:
            sum_spec = np.sum(self.cube, axis=(1,2))

        else:
            if self.cube_err is not None:
                sum_spec = np.sum(self.cube_err, axis=0)
            else:
                print('ERROR: no error cube provided')
                return None

        spec_obj = spectrum(sum_spec, self.wave, z=None, obj_name=self.object)
        if err==False:
            self.col_cube_spec = spec_obj
        
        if save:
            outname = self.filename.split('.fits')[0]+'_collpase_spec.fits'
            if err:
                outname = outname[0:-5]+'_err.fits'
            hdr_comment = 'BUILT collpased spectrum of data cube'
            spec_obj.save_fits(outname, hdr_comment=hdr_comment)

        return spec_obj

    def extract_spectrum(self, RA, DEC, apert_rad, err=False):
        
        if isinstance(RA, float) and isinstance(DEC, float):
            c = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, frame='icrs')
        else:
            print('ERROR: RA and DEC must be in degrees')
            return None
        
        if not isinstance(apert_rad, float):
            print('ERROR: apert_rad must be in degrees')
            return None

        grid_sep = self.coord_grid.separation(c)
        ap_inds = np.where(grid_sep < apert_rad*u.degree)
        
        if not err:
            ap_cube = self.cube[ap_inds]
        else:
            if self.cube_err is not None:
                ap_cube = self.cube_err[ap_inds]
            else:
                print('ERROR: no error cube provided')
                return None
            
        ext_spec = np.sum(ap_cube, axis=0)

        return ext_spec
    
    def extract_star_spec(self):
        if self.col_cube_im is None:
            col_cube = self.collapse_frame()
        
        sources_df = imu.find_stars(self.col_cube_im, star_thres=10., 
                                    num_bright_stars=1, star_fwhm=8.0)
        
        sources_df = imu.measure_star_params(self.col_cube_im,
                                             sources_df.copy())
        
        #star_cent_ra = sources_df.iloc[0]['RA']
        #star_cent_dec = sources_df.iloc[0]['DEC']
        #star_ap = sources_df.iloc[0]['fwhm']
        #star_spec = self.extract_spectrum(star_cent_ra, star_cent_dec,
        #                                    star_ap)
        #star_spec_err = self.extract_spectrum(star_cent_ra, star_cent_dec,
        #                                    star_ap, err=True)
        
        return sources_df
        

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
