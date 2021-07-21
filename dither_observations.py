#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 15:28:36 2021

@author: Briana
"""

import pandas as pd
import numpy as np
import sys

from astropy import units as u
from astropy.io import fits
from astropy.coordinates import Angle

import interpolate_IFU
import guider_observations as go
import IFU_spectrum as ifu_spec
import VP_fits_frame as vpf


class dither_observation():

    def __init__(self, VP_frames, dither_group_id=0,
                 dith_file='VP_config/dith_vp_6subdither.csv'):

        self.VP_frames = VP_frames

        self.dith_order_lis = np.ones(len(VP_frames))
        for f in range(len(self.VP_frames)):
            if not isinstance(self.VP_frames[f], vpf.VP_fits_frame):
                sys.exit('Must provide list of VP_fits_frame objects for dither set')
            else:
                self.dith_order_lis[f] = self.VP_frames[f].dith_num
                if dither_group_id is not None:
                    self.VP_frames[f].dither_group_id = dither_group_id

        self.dither_group_id = dither_group_id
        self.dith_df = pd.read_csv(dith_file, skiprows=2)

        self.wave = None
        self.wave_start = None
        self.wave_delta = None
        self.master_spec = None
        self.master_err_spec = None
        self.master_fib_df = None

        self.data_cube = None
        self.data_err_cube = None
        self.cube_wcs = None

        self.field_RA = None
        self.field_DEC = None

        print('BUILD '+str(len(self.VP_frames))+' dither observation: [DITHOBS:'+str(self.dither_group_id)+']')

    def normalize_dithers(self, guide_obs, star_thres=10., num_bright_stars=10,
                          star_fwhm=8.0, fwhm_lim=(0.5, 10), mag_lim=10):

        if isinstance(guide_obs, go.guider_observations):
  
            print(' [DITHOBS:'+str(self.dither_group_id)+'] build normalized dithers')

            # check if matched each fits image has matched guider frames
            obs_guide_lis = []
            for f in range(len(self.VP_frames)):
                if self.VP_frames[f].guide_match is False:
                    self.VP_frames[f].match_guider_frames(guide_obs)
                obs_guide_lis.append(self.VP_frames[f].guider_ind)
            obs_guide_lis = np.hstack(obs_guide_lis)
            
            if len(obs_guide_lis) > 0:

                # find the RA/DEC dither shift in pixels in guider cam frames
                self.dith_df['RA_pix_shift'] = self.dith_df['RA_shift']/guide_obs.guider_ps
                self.dith_df['DEC_pix_shift'] = self.dith_df['DEC_shift']/guide_obs.guider_ps
    
                # find a guider image to use as reference for stars
                print('GUIDE', obs_guide_lis)
                ref_sources_df, ref_guide_ind = guide_obs.find_ref_guide_frame(obs_guide_lis, star_thres=star_thres, 
                                                num_bright_stars=num_bright_stars, star_fwhm=star_fwhm, 
                                                fwhm_lim=fwhm_lim, mag_lim=mag_lim)
    
                guide_stars_lis = []
                for f in self.VP_frames:
                    guide_ind_lis = f.guider_ind
                    frame_dith_num = f.dith_num
    
                    # update ref_sources_df with the correct dither coordinates
                    shift_df = self.dith_df[self.dith_df['dith_num'] == frame_dith_num]
                    x_mod = shift_df['RA_pix_shift'].values[0]
                    y_mod = shift_df['DEC_pix_shift'].values[0]
                    mod_ref_sources_df = ref_sources_df.copy()
                    mod_ref_sources_df['xcentroid'] = ref_sources_df['xcentroid']-x_mod
                    mod_ref_sources_df['ycentroid'] = ref_sources_df['ycentroid']-y_mod
    
                    for g in guide_ind_lis:
                        sources_fit = guide_obs.measure_guide_star_params(g, mod_ref_sources_df)
                        sources_fit = guide_obs.flag_stars(sources_fit.copy(),
                                                           fwhm_lim=fwhm_lim,
                                                           mag_lim=mag_lim)
                        sources_fit['dith_num'] = frame_dith_num
                        sources_fit['guide_ind'] = g
    
                        good_sources_fit = sources_fit[sources_fit['bad_flag'] == False]
                        guide_stars_lis.append(good_sources_fit)
    
                guide_stars_df = pd.concat(guide_stars_lis).reset_index()
                dith_stars_df = guide_stars_df.groupby(by = ['dith_num', 'id'],
                                                       as_index=False).mean()
    
                dith_star_count = dith_stars_df[['id', 'dith_num']].groupby(by='dith_num', as_index=False).count()
                dith_star_count.rename(columns={'id': 'num_stars_used'},
                                       inplace=True)
    
                dith_flux_sum = dith_stars_df[['dith_num', 'flux_fit']].groupby(by='dith_num', as_index=False).sum()
                dith_flux_sum['flux_norm'] = dith_flux_sum['flux_fit']/dith_flux_sum['flux_fit'].max()
    
                dith_fwhm_avg = dith_stars_df[['dith_num', 'fwhm(arcseconds)']].groupby(by='dith_num', as_index=False).mean()
    
                dith_norm_df1 = dith_flux_sum.merge(dith_fwhm_avg, on='dith_num',
                                                    how='outer')
                dith_norm_df = dith_norm_df1.merge(dith_star_count, on='dith_num',
                                                   how='outer')
    
                for i in self.VP_frames:
                    frame_dith_num = i.dith_num
                    see_val = dith_norm_df[dith_norm_df['dith_num']==frame_dith_num]['fwhm(arcseconds)'].values[0]
                    norm_val = dith_norm_df[dith_norm_df['dith_num']==frame_dith_num]['flux_norm'].values[0]
     
                    i.seeing = see_val
                    i.dithnorm = norm_val
                    i.hdr['SEEING'] = see_val
                    i.hdr['DITHNORM'] = norm_val
                    dith_norm_dat = i.dat*norm_val
                    i.dat = dith_norm_dat
    
                    i.build_new_extension('dithnorm',
                                          'normalize dither spec from guide star flux')
                    i.fits_ext = 'dithnorm'
            
            else:
                print('WARNING: no guider frames found. [DITHOBS:'+str(self.dither_group_id)+'] will NOT be normalized')

        else:
            print('guide_obs must be a guider_observations class object')
            return None

    def build_common_wavesol(self):

        print(' [DITHOBS:'+str(self.dither_group_id)+'] build common wavelength solution')

        # establish the wavelength solution of the first dither for all dithers
        dith1_obj = self.VP_frames[np.where(self.dith_order_lis == 1)[0][0]]
        self.wave = dith1_obj.wave
        self.wave_start = dith1_obj.wave_start
        self.wave_delta = dith1_obj.wave_delta

        for i in range(len(self.VP_frames)):
            frame_wave = self.VP_frames[i].wave

            if not np.array_equal(frame_wave, self.wave):
                print('Fixing dither', self.VP_frames[i].dith_num,
                      'to common wavelength grid')
                self.VP_frames[i].wave = self.wave
                old_dat = self.VP_frames[i].dat

                new_spec_lis = []
                for i in range(np.shape(old_dat)[0]):
                    spec_obj = ifu_spec.IFU_spectrum(old_dat[i], frame_wave)
                    spec_obj.new_wave_sol(self.wave)
                    new_spec_lis.append(spec_obj.spec)

                new_dat = np.vstack(new_spec_lis)

                self.VP_frames[i].dat = new_dat
                self.VP_frames[i].hdr['CRVAL1'] = dith1_obj.wave_start
                self.VP_frames[i].hdr['CDELT1'] = dith1_obj.wave_delta

                new_ext_name = 'comwave'
                hdr_comment = 'interp. to common wavelength grid'
                self.VP_frames[i].build_new_extension(new_ext_name, hdr_comment)
                self.VP_frames[i].fits_ext = new_ext_name

    def build_master_fiber_files(self):

        if not isinstance(self.wave, np.ndarray):
            self.build_common_wavesol()
            
        print(' [DITHOBS:'+str(self.dither_group_id)+'] build master dither set files')

        dith1_obj = self.VP_frames[np.where(self.dith_order_lis == 1)[0][0]]
        self.field_RA = dith1_obj.RA
        self.field_DEC = dith1_obj.DEC

        dat_lis = []
        err_lis = []
        fib_df_lis = []
        # shift fiber RA,DEC by dither file offsets
        for d in self.dith_order_lis:
            dith_obj = self.VP_frames[np.where(self.dith_order_lis == d)[0][0]]

            RA_dith_shift_as = self.dith_df[self.dith_df['dith_num'] == d][
                    'RA_shift'].values[0]

            DEC_dith_shift_as = self.dith_df[self.dith_df['dith_num'] == d][
                    'DEC_shift'].values[0]
            RA_dith_shift_deg = RA_dith_shift_as/3600
            DEC_dith_shift_deg = DEC_dith_shift_as/3600

            dat_lis.append(dith_obj.dat)
            err_lis.append(dith_obj.dat_err)

            cen_ra_shift = dith_obj.fib_df['RA_offset'].values/3600
            cen_dec_shift = dith_obj.fib_df['DEC_offset'].values/3600

            dith_obj.fib_df['RA'] = self.field_RA + cen_ra_shift + \
                RA_dith_shift_deg
            dith_obj.fib_df['DEC'] = self.field_DEC + cen_dec_shift + \
                DEC_dith_shift_deg
            fib_df_lis.append(dith_obj.fib_df)

        self.master_spec = np.vstack(dat_lis)
        self.master_err_spec = np.vstack(err_lis)

        fib_df = pd.concat(fib_df_lis)
        self.master_fib_df = fib_df.reset_index(drop=True)

    def make_data_cube(self, grid=(0, 0, 0, 0)):

        if not isinstance(self.master_spec, np.ndarray):
            self.build_master_fiber_files()

        print(' [DITHOBS:'+str(self.dither_group_id)+'] build data cube and error cube')

        fiberd_as = self.VP_frames[0].fib_df.iloc[0]['fiberd']  # in arcseconds
        # convert all arcsecond units to degrees
        fiberd = Angle(fiberd_as*u.arcsecond).degree

        fib_RA = self.master_fib_df['RA'].values
        fib_DEC = self.master_fib_df['DEC'].values

        regrid_size = fiberd/2.0
        kern_sig = fiberd
        max_radius = fiberd*5.0
        self.interp_class = interpolate_IFU.fibers_to_grid(fib_RA,
                                                           fib_DEC, fiberd,
                                                           regrid_size,
                                                           max_radius,
                                                           kern_sig)

        # if all input grid values are the same then assume
        # no grid is defined and use coordinate list to define new grid
        if len(set(list(grid))) == 1:
            xmin = self.master_fib_df['RA'].min()
            xmax = self.master_fib_df['RA'].max()
            ymin = self.master_fib_df['DEC'].min()
            ymax = self.master_fib_df['DEC'].max()
            grid = (xmin, xmax, ymin, ymax)

        x_grid, y_grid = self.interp_class.build_new_grid(grid=grid)

        wave_frame_lis = []
        wave_frame_err_lis = []
        for i in range(len(self.wave)):
            wave_frame, wave_err_frame = self.interp_class.shepards_kernal(
                    self.master_spec[:, i], self.master_err_spec[:, i])
            wave_frame_lis.append(wave_frame)
            wave_frame_err_lis.append(wave_err_frame)

        self.data_cube = np.dstack(wave_frame_lis)
        self.data_err_cube = np.dstack(wave_frame_err_lis)

        wave_wcs_dict = {'CUNIT3': 'Angstrom',
                         'CTYPE3': 'Angstrom',
                         'CRPIX3': 1,
                         'CRVAL3': self.wave_start,
                         'CDELT3': self.wave_delta}

        self.cube_wcs = self.interp_class.build_wcs(wave_dict=wave_wcs_dict)

    def write_data_cube(self):

        if not isinstance(self.data_cube, np.ndarray):
            self.make_data_cube()

        print(' [DITHOBS:'+str(self.dither_group_id)+'] build fits cube files')

        # find mean of header values for combined dithers
        wcs_hdr = self.cube_wcs.to_header()

        dith1_obj = self.VP_frames[np.where(self.dith_order_lis == 1)[0][0]]
        dith1_hdr = dith1_obj.hdr

        wcs_hdr['NUMDITH'] = (len(self.VP_frames),
                              'Number of combined dithers')
        wcs_hdr['REGRID'] = (self.interp_class.regrid_size,
                             'Regrid size set for interpolation (deg)')
        wcs_hdr['KERNSIG'] = (self.interp_class.kern_sig,
                              'Sigma for gaussian interpolation kernal (deg)')
        wcs_hdr['MAXRAD'] = (self.interp_class.max_dist,
                             'Max radius pixel center (deg)')
        wcs_hdr['RA'] = (self.field_RA, 'RA of object for dither 1 (deg)')
        wcs_hdr['DEC'] = (self.field_DEC, 'DEC of object for dither 1 (deg)')
        wcs_hdr['EQUINOX'] = dith1_hdr['EQUINOX']
        wcs_hdr['OBJECT'] = dith1_obj.object
        wcs_hdr['DATE-OBS'] = (dith1_obj.obs_datetime.strftime('%Y%m%dT%H%M%S'), 'Obs date for dither 1')
        wcs_hdr['EXPTIME'] = (dith1_obj.exptime,
                              'Average dither exposure time')
        wcs_hdr['FILEXT'] = (dith1_obj.fits_ext, 'fits extention used')

        for d in self.dith_order_lis:
            dith_obj = self.VP_frames[np.where(self.dith_order_lis == d)[0][0]]
            wcs_hdr['AIRMAS'+str(int(d))] = dith_obj.airmass
            wcs_hdr['DITHNOR'+str(int(d))] = dith_obj.dithnorm
            wcs_hdr['SEEING'+str(int(d))] = dith_obj.seeing
            wcs_hdr['DITHFIL'+str(int(d))] = dith_obj.filename
            wcs_hdr['FILEXT'+str(int(d))] = dith_obj.fits_ext

        wcs_hdr['COMMENT'] = 'DATA CUBE built from DITHFIL# files'
        wcs_hdr['COMMENT'] = 'using the fits extension FILEXT'

        self.wcs_hdr = wcs_hdr

        # reshape data cube so dimenstions read correctly by fits
        # fits uses Fortran convention (column-major order) (z, y, x)
        # python uses (y, x, z) so need to swap them
        fits_cube = np.swapaxes(np.swapaxes(self.data_cube, 2, 0), 1, 2)
        hdu_new = fits.PrimaryHDU(fits_cube, header=wcs_hdr)

        outname = self.VP_frames[0].filename.split('.')[-2] + \
            '_data_cube_'+str(int(self.dither_group_id))+'.fits'

        hdu_new.writeto(outname, overwrite=True)

        hdu_err = fits.open(outname)
        hdu_err[0].header['FILEXT'] = dith1_obj.fits_err_ext
        hdu_err[0].header['COMMENT'] = 'ERROR DATA CUBE built from DITHFIL#'
        hdu_err[0].header['COMMENT'] = 'files using the fits error extension FILEXT'
        fits_cube_err = np.swapaxes(np.swapaxes(self.data_err_cube, 2, 0), 1, 2)
        hdu_err[0].data = fits_cube_err
        outname_err = outname.split('.fits')[0]+'_err.fits'
        hdu_err.writeto(outname_err, overwrite=True)
        hdu_err.close()

    # fib_ind (list/array)(optional,
    # default will sum all fibers): list of fiber indices to sum and plot
    def build_frame_sum_spec(self, fib_inds=[], z=np.NaN, plot=False):
        if len(fib_inds) == 0:
            spec = np.sum(self.dat, axis=0)
        elif len(fib_inds) == 1:
            spec = self.dat[fib_inds[0]]
        else:
            spec = np.sum(self.dat[fib_inds, :], axis=0)

        sum_spec = ifu_spec.IFU_spectrum(spec, self.wave, z=z)

        if plot:
            sum_spec.plot_spec(spec_units='Electrons per second')

        return sum_spec
