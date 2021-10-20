#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 11:07:41 2021

@author: Briana
"""

import pandas as pd
import numpy as np
import numpy.ma as ma
from scipy import interpolate, signal
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS, utils
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.convolution import Gaussian1DKernel, convolve
from astropy.modeling import models, fitting

import image_utils as imu
from IFU_spectrum import spectrum


class Cube():

    def __init__(self, cube_file):

        try:
            hdu = fits.open(cube_file)
        except FileNotFoundError():
            raise FileNotFoundError('Can not Find cube_file: '+str(cube_file))

        self.filename = cube_file
        self.hdr = hdu[0].header
        self.cube = hdu[0].data
        hdu.close()

        try:
            err_cube_file = cube_file[0:-5]+'_err.fits'
            hdu_err = fits.open(err_cube_file)
            self.cube_err = hdu_err[0].data
            hdu_err.close()
        except FileNotFoundError:
            raise FileNotFoundError('Can not find cube error file: '+str(cube_file))

        self.object = self.hdr['OBJECT']
        self.RA = self.hdr['RA']
        self.DEC = self.hdr['DEC']
        self.equinox = self.hdr['EQUINOX']

        self.units = self.hdr['UNITS']
        self.flux_cal = bool(self.hdr['FLUXCAL'])
        self.sens_curve = self.hdr['CALFILE']

        self.wcs = WCS(self.hdr)
        self.num_pix = np.shape(self.cube)[1] * np.shape(self.cube)[2]
        self.wcs_2d = WCS(self.hdr, naxis=[1, 2])
        self.x_grid, self.y_grid = np.meshgrid(np.arange(np.shape(self.cube)[2]),
                                               np.arange(np.shape(self.cube)[1]))
        self.coord_grid = utils.pixel_to_skycoord(self.x_grid, self.y_grid,
                                                  self.wcs_2d)
        self.grat_res = self.hdr['GRATRES']
        self.pix_scale = self.hdr['REGRID']
        self.num_dith = self.hdr['NUMDITH']
        self.wave = self.hdr['CRVAL3'] + ((np.arange(self.hdr['NAXIS3']) * self.hdr['CDELT3']))
        
        self.col_cube_im = None
        self.col_cube_spec = None
        
        print('BUILD cube object: [CUBE:'+str(self.object)+']')


    def build_2D_hdr():
        # relies on the assumption the header for the data cube
        # is formatted so that the wcs information is written right
        # below the basic header keywords so it looks for the end of the
        # WCS keywords by looking for 'RADESYS'='ICRS'
        # and copies everything below
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
            # if no wave range provide collapse entire cube into 2d frame
            wave_inds = np.arange(len(self.wave))
            wave_name = 'all'
            
        print('    [CUBE] build collapsed frame: '+wave_name)

        if not err:
            col_frame = np.nansum(self.cube[wave_inds], axis=0)

        else:
            col_frame = np.nansum(self.cube_err[wave_inds], axis=0)

        if (wave_name == 'all') and (err==False):
            self.col_cube_im = col_frame

        if save:
            #add giving it the cube header but only 2D WCS
            hdr_im = self.build_2D_hdr()
            hdu = fits.PrimaryHDU(col_frame, header=hdr_im)            
            outname = self.filename.split('.fits')[0]+'_collpasecube_'+wave_name+'.fits'
            if err:
                outname = outname[0:-5]+'_err.fits'
            hdu.writeto(outname, overwrite=True)
            print('    [CUBE] save collpased frame: '+outname)
   
        return col_frame

    def collapse_spectrum(self, err=False, save=False):
        
        print('    [CUBE] build collapsed spectrum')

        if not err:
            sum_spec = np.nansum(self.cube, axis=(1,2))

        else:
            sum_spec = np.nansum(self.cube_err, axis=0)

        spec_obj = spectrum(sum_spec, self.wave, z=None, obj_name=self.object)
        if err==False:
            self.col_cube_spec = spec_obj

        if save:
            outname = self.filename.split('.fits')[0]+'_collpase_spec.fits'
            if err:
                outname = outname[0:-5]+'_err.fits'
            hdr_comment = 'BUILT collpased spectrum of data cube'
            spec_obj.save_fits(outname, hdr_comment=hdr_comment)
            print('    [CUBE] save collpased spectrum: '+outname)

        return spec_obj

    def extract_spectrum(self, RA, DEC, apert_rad, err=False):
        
        print('    [CUBE] extract spectrum')

        if isinstance(RA, float) and isinstance(DEC, float):
            c = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, frame='icrs')
        else:
            raise ValueError('RA and DEC must be in degrees')

        if not isinstance(apert_rad, float):
            raise ValueError('apert_rad must be in degrees')

        grid_sep = self.coord_grid.separation(c)
        ap_inds = np.where(grid_sep < apert_rad*u.degree)

        if not err:
            ap_cube = self.cube[ap_inds]
        else:
            ap_cube = self.cube_err[ap_inds]

        ext_spec = np.nansum(ap_cube, axis=0)

        return ext_spec

    def extract_standard_star(self, psf_extract=True, fix_ap=False,
                              fix_ap_rad=30):
        """Extracts a standard star spectrum from the data cube.

        Uses extract_spectrum to find standard star center and extract.
        Uses either a Moffat PSF model to get total star spectrum
        or uses a 3*sigma aperature or a fixed aperture (fix_ap_rad).

        Args:
            psf_extract (boolean): extract star using moffat psf model,
                defaults to True
            fix_ap (boolean): If True uses fix_ap_rad to set extraction 
                aperture, else uses a 3*sigma aperture,
                defaults to False  
            fix_ap_rad (int, float): fixed extraction aperture in arcseconds,
                defaults to 30.0
        Returns:
            ndarray: star spectrum
            ndarray: star error spectrum
        Raises:
            ValueError: if no stars are found in the data cube
        """
        print('    [CUBE] extract standard star spectrum')
        
        if self.col_cube_im is None:
            self.col_cube_im = self.collapse_frame()
        
        if psf_extract:
            star_spec = np.ones(len(self.wave))
            star_spec_err = np.ones(len(self.wave))
            
            mof_mod = imu.fit_moffat_psf(self.col_cube_im)
            sum_weights = np.sum(mof_mod)

            for i in range(len(self.wave)):
                col_im = self.cube[i]
                col_err = self.cube_err[i]

                weighted_col_im = col_im * mof_mod
                weighted_col_err = col_err * mof_mod

                fit_flux = np.nansum(weighted_col_im)/sum_weights
                fit_err = np.nansum(weighted_col_err)/sum_weights
            
                star_spec[i] = fit_flux
                star_spec_err[i] = fit_err

        else:

            sources_df = imu.find_stars(self.col_cube_im, star_thres=10.,
                                        num_bright_stars=1, star_fwhm=8.0)
            if sources_df is None:
                raise ValueError('No stars found in frame')

            sources_df = imu.measure_star_params(self.col_cube_im,
                                                 sources_df.copy())

            star_cent_x = sources_df.iloc[0]['xcentroid_fit']
            star_cent_y = sources_df.iloc[0]['ycentroid_fit']
            star_cent_ra = star_cent_x*self.hdr['CDELT1'] + self.hdr['CRVAL1']
            star_cent_dec = star_cent_y*self.hdr['CDELT2'] + self.hdr['CRVAL2']

            star_fwhm_pix = sources_df.iloc[0]['fwhm(pixels)']
            star_sig_pix = star_fwhm_pix*2.355

            star_sig_deg = star_sig_pix*self.hdr['CDELT1']

            if fix_ap:
                ap_ext = fix_ap_rad
            else:
                ap_ext = 3*star_sig_deg

            star_spec = self.extract_spectrum(star_cent_ra, star_cent_dec,
                                              ap_ext)
            star_spec_err = self.extract_spectrum(star_cent_ra, star_cent_dec,
                                                  ap_ext, err=True)
        return star_spec, star_spec_err

    def match_calspec_standard(self, ra, dec, max_sep=0.014, per_err=5.0):
        """Matches standard with known spec from stsci standard repository

        Args:
            ra, dec (int, float): RA and DEC of standard star to match
            max_sep (int_float): maximum separation (degrees) ra/dec can be
                from known standard coordinates to be considered a match,
                defaults to 0.014  
            per_err (int, float): percent error of the spectrum to use as error
                spectrum if one is not found in the repository,
                defaults to 5.0%
        Returns:
            ndarray: star spectrum (None if not found)
            ndarray: wavelength solution (None if not found)
            ndarray: star error spectrum (None if not found)
        Raises:
            ValueError: if ra, dec, max_sep not given in degrees
            ValueError: if per_err used and not percent between 0-100
        """
        
        print('    [CUBE] match CALSPEC standard')

        if isinstance(ra, (float, int)) and isinstance(dec, (float, int)):
            stand_df = pd.read_csv('standard_stars/standard_stars.csv')
            cat_ra = stand_df['RA'].values
            cat_dec = stand_df['DEC'].values
            cat_coord = SkyCoord(cat_ra, cat_dec, unit='deg', frame='icrs')

            c = SkyCoord(ra, dec, unit='deg', frame='icrs')
            match_ind, match_sep, match3d = match_coordinates_sky(c, cat_coord)
            match_sep = match_sep.value[0]

            if isinstance(max_sep, (float, int)):

                if match_sep < max_sep:
                    match_df = stand_df.iloc[match_ind]
                    match_file = 'standard_stars/'+match_df['data_source']+'/'+match_df['spec_file']
                    print('    [CUBE] '+self.object+' matched with file: '+match_file)
                    print(match_df)
                    print(match_df['data_source'])

                    if match_df['data_source'] == 'stsci_calspec':
                        hdu = fits.open(match_file)
                        dat = hdu[1].data

                        spec = dat.field('Flux')
                        wave = dat.field('WAVELENGTH')

                        stat_err = dat.field('STATERROR')
                        sys_err = dat.field('SYSERROR')
                        spec_err = np.sqrt((stat_err**2) + (sys_err**2))
                    else:
                        print('ELSE')
                        if (per_err > 0) and (per_err < 100):
                            dat_df = pd.read_csv(match_file)
                            spec = dat_df['flux_flam']
                            print(np.shape(spec))
                            wave = dat_df['wave']
                            spec_err = spec*(per_err/100)

                            return spec, wave, spec_err

                        else:
                            raise ValueError('per_err must be a percent (0-100)')

                else:
                    return None, None, None

            else:
                raise ValueError('max_sep must be given in degrees')

        else:
            raise ValueError('ra and dec must be given in degrees')

    def build_sensitiviy_curve(self, cal_spec=None, cal_wave=None,
                               cal_err=None, cal_per_err=5.0, save=True,
                               plot=False):
        
        print('    [CUBE] build sensitivity Curve for '+str(self.object))
        
        # find lit spectrum to compare
        if cal_spec is None:
            cal_spec, cal_wave, cal_err = self.match_calspec_standard(self.RA,
                                                                        self.DEC,
                                                                        per_err=cal_per_err)
            if cal_spec is None:
                raise ValueError('Could not find standard match in Calspec STIS catalog of standards. \
                                 Please provide a cal_spec and cal_wave for this standard')

        elif isinstance(cal_spec, (np.ndarray, list)):

            if isinstance(cal_wave, (np.ndarray, list)):
                spec_shape = np.shape(cal_spec)
                wave_shape = np.shape(cal_wave)

                if spec_shape == wave_shape:

                    if isinstance(cal_err, (np.ndarray, list)):
                        err_shape = np.shape(cal_err)

                        if err_shape != spec_shape:
                            # add checks to these for reasonable units/in
                            # wave range of data
                            raise ValueError('The error spectrum must match the shape\
                                             of cal_spec and cal_wave')
                    else:

                        if isinstance(cal_per_err, (int, float)):

                            if (cal_per_err > 0) and (cal_per_err < 100):
                                print('WARNING: no error spectrum found for known standard.\
                                      Assuming known standard spec error of', cal_per_err,
                                      'can assign with cal_per_err')
                                cal_err = cal_spec*(cal_per_err/100)

                            else:
                                raise ValueError('cal_per_err must be a percent (0-100)')

                        else:
                            raise ValueError('cal_per_err must be an int or float between 0-100')

                else:
                    raise ValueError('The cal_spec must match the shape as cal_wave')

            else:
                raise ValueError('cal_wave must be array-like')

        else:
            raise ValueError('If providing known star spectrum cal_spec and cal_wave must be array-like')

        # call extract spectrum to build spectrum of star
        cube_spec, cube_err = self.extract_standard_star()

        # convolve with resolution of the spectrum
        # Create kernel
        # wl dependend kernel
        g_kern = Gaussian1DKernel(stddev=self.grat_res)
        cal_spec_conv = convolve(cal_spec, g_kern)
        cal_err_conv = convolve(cal_err, g_kern)

        # interpolate the known spec to cube wavelength
        f_cal = interpolate.interp1d(x=cal_wave, y=cal_spec_conv)
        cal_spec_interp = f_cal(self.wave)

        f_cal_err = interpolate.interp1d(x=cal_wave, y=cal_err_conv)
        cal_err_interp = f_cal_err(self.wave)

        # mask out the Balmer lines
        balm_wave = [6563, 4861, 4340, 4102, 3970, 3889, 3835]
        ma_cal_spec = ma.array(cal_spec_interp)
        ma_wave = ma.array(self.wave)
        ma_cal_err = ma.array(cal_err_interp)

        for i in balm_wave:
            if (i > np.amin(self.wave)) & (i < np.amax(self.wave)):
                # cut out region around line
                wave_fit_inds = np.where((self.wave > (i-50)) &
                                         (self.wave < (i+50)))[0]
                wave_fit = self.wave[wave_fit_inds]
                spec_fit = cal_spec_interp[wave_fit_inds]*1e13
                err_fit = cal_err_interp[wave_fit_inds]*1e13

                # define continuum region
                cont_end = int(len(spec_fit)*0.1)
                spec_sort = np.append(spec_fit[0:cont_end],
                                      spec_fit[-cont_end::])
                cont_val = np.median(spec_sort)
                poly_cont = models.Polynomial1D(1, c0=cont_val, c1=0)

                # define absorption line
                g_bounds = {'amplitude': (-1e5, 0), 'mean': (i-5, i+5),
                            'stddev': (5.3, 40)}
                g1 = models.Gaussian1D(amplitude=-1, mean=i, stddev=5.3,
                                       bounds=g_bounds)
                fitter = fitting.LevMarLSQFitter()
                fit = fitter(g1+poly_cont, wave_fit, spec_fit, weights=err_fit)
                fit_width = 3*fit.stddev_0.value

                # masking absorption line
                mask_inds = np.where((self.wave <= i+fit_width) &
                                     (self.wave >= i-fit_width))
                ma_cal_spec[mask_inds] = ma.masked
                ma_wave[mask_inds] = ma.masked
                ma_cal_err[mask_inds] = ma.masked

        balm_mask = ma_wave.mask
        ma_cube_spec = ma.masked_array(cube_spec, mask=balm_mask)

        ma_sens_func = np.divide(ma_cube_spec, ma_cal_spec)
        spl = interpolate.UnivariateSpline(ma_wave.compressed(),
                                           ma_sens_func.compressed(), k=1)
        sens_func_spl = spl(self.wave)

        sens_func = signal.medfilt(sens_func_spl, kernel_size=51)
        self.sens_func = sens_func

        if save:
            save_df = pd.DataFrame({'wave': self.wave,
                                    'sens_func': sens_func,
                                    'cube_spec': cube_spec,
                                    'cal_spec': cal_spec_interp,
                                    'balmer_mask': balm_mask})
            sens_outfile = self.filename[0:-5]+'_SENS_CURV'
            save_df.to_csv(sens_outfile+'.csv', index=False)
            print('    [CUBE] save sensitivity curve: '+sens_outfile)

        if plot:
            fig, ax = plt.subplots(3, 1, figsize=(12, 10))

            ax[0].plot(self.wave, cube_spec, color='blue', lw=2,
                       label='cube spec')
            ax[0].plot(ma_wave, ma_cube_spec, color='red', lw=1,
                       label='masked cube spec')
            ax[0].set_ylabel('e-/s', fontsize=15)

            ax[1].plot(self.wave, cal_spec_interp, color='green', lw=2,
                       label='cal. spec conv')
            ax[1].plot(ma_wave, ma_cal_spec, color='red', lw=1,
                       label='masked cal. spec')
            ax[1].set_ylabel('Flux (erg/s/cm^2/A)', fontsize=15)
            
            ax[2].plot(self.wave, sens_func_spl, color='grey', lw=3,
                       label='sens func spline')
            ax[2].plot(ma_wave, ma_sens_func, color='red', lw=1.5,
                       label='masked sens func')
            ax[2].plot(self.wave, sens_func, color='black', lw=1.5,
                       label='smoothed sens func')
            ax[2].set_ylabel('e-/s/Flux', fontsize=15)

            ax[2].set_xlabel('Wavelength(A)', fontsize=15)
            ax[0].legend(fontsize=20)
            ax[1].legend(fontsize=20)
            ax[2].legend(fontsize=20)
            if save:
                plt.savefig(sens_outfile+'.png')
            plt.show()

        return sens_func

    def save_new_cube(self, outname_tag='_new', err=False, overwrite=True):
        if not err:
            hdu_new = fits.PrimaryHDU(self.cube, header=self.hdr)
            out_mess = '    [CUBE]: save '+outname_tag+' cube '
        else:
            hdu_new = fits.PrimaryHDU(self.cube_err, header=self.hdr)
            out_mess = '    [CUBE]: save '+outname_tag+' error cube '

        outname = self.filename[0:-5]+str(outname_tag)+'.fits'
        hdu_new.writeto(outname, overwrite=overwrite)
        print(out_mess + outname)

    def flux_calibrate(self, sens_curve_file, save=True):

        print('Flux Calibrate Data Cube: [CUBE:'+str(self.object)+']')

        # read in sensitivity curve
        try:
            sens_df = pd.read_csv(sens_curve_file)
            sens_wave = sens_df['wave']
            sens_func = sens_df['sens_func']
            
        except FileNotFoundError:
            raise FileNotFoundError('could not find: '+str(sens_curve_file))
            
        except KeyError:
            raise KeyError('need to have columns: wave, sens_func')

        # check sense curve covers wavelength range of data

        # interpolate sensitivity function so on same wavelength grid
        f_sens = interpolate.interp1d(x=sens_wave, y=sens_func,
                                      fill_value="extrapolate")
        self.sens_curve = f_sens(self.wave)

        # flux calibrate cube and error cube
        fluxCalib_cube = self.cube.copy()
        fluxCalib_cube_err = self.cube_err.copy()
        for y in range(np.shape(self.cube)[1]):
            for x in range(np.shape(self.cube)[2]):
                fluxCalib_cube[:,y, x] = self.cube[:,y, x] / self.sens_curve
                fluxCalib_cube_err[:,y, x] = self.cube_err[:,y, x] / self.sens_curve

        self.cube = fluxCalib_cube
        self.cube_err = fluxCalib_cube_err
        self.hdr['FLUXCAL'] = 'True'
        self.hdr['CALFILE'] = str(sens_curve_file)
        self.hdr['UNITS'] = 'egs/s/cm^2/A'
        self.flux_calib = True

        if save:
            self.save_new_cube(outname_tag='_FluxCal')
            self.save_new_cube(outname_tag='_FluxCal', err=True)

    def regrid_cube(self, apert_rad):
        # may not make sense here b/c would need master fiber data
        # could write this into the object
        return None
