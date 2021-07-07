#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 15:28:36 2021

@author: Briana
"""

import pandas as pd
import numpy as np
import warnings

from astropy import units as u
from astropy.io import fits
from astropy.coordinates import Angle

import interpolate_IFU
import guider_observations as go
import IFU_spectrum as ifu_spec
import VP_fits_frame as vpf


class dither_observation():

    def __init__(self, VP_frames, dither_group_id=None,
                 dith_file='VP_config/dith_vp_6subdither.csv'):

        self.VP_frames = VP_frames

        self.dith_order_lis = np.ones(len(VP_frames))
        for f in range(len(self.VP_frames)):
            if not isinstance(self.VP_frames[f], vpf.VP_fits_frame):
                print('Must provide list of VP_fits_frame objects for \
                      dither set')
                return None
            else:
                self.dith_order_lis[f] = self.VP_frames[f].dith_num
                if dither_group_id is not None:
                    self.VP_frames[f].dither_group_id = dither_group_id

        self.dither_group_id = dither_group_id
        self.dith_df = pd.read_csv(dith_file, skiprows=2)

        self.wave = None
        self.master_spec = None
        self.master_err_spec = None
        self.master_fib_df = None

        self.data_cube = None
        self.data_err_cube = None

    def normalize_dithers(self, guide_obs, star_thres=10., num_bright_stars=10,
                          star_fwhm=8.0, fwhm_lim=(0.5, 10), mag_lim=10):

        if isinstance(guide_obs, go.guider_observations):

            # check if matched each fits image has matched guider frames
            obs_guide_lis = []
            for f in range(len(self.VP_frames)):
                if self.VP_frames[f].guide_match is None:
                    print('MATCHING GUDIE FRAMES')
                    self.VP_frames[f].match_guider_frames(guide_obs)
                obs_guide_lis.append(self.VP_frames[f].guider_ind)
            obs_guide_lis = np.hstack(obs_guide_lis)

            # find the RA/DEC dither shift in pixels in guider cam frames
            self.dith_df['RA_pix_shift'] = self.dith_df['RA_shift']/guide_obs.guider_ps
            self.dith_df['DEC_pix_shift'] = self.dith_df['DEC_shift']/guide_obs.guider_ps

            # find a guider image to use as reference for stars
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

                i.fits_ext = 'dithnorm'
                i.build_new_extension('dithnorm',
                                      'normalize dither spec from guide star flux')

        else:
            print('guide_obs must be a guider_observations class object')
            return None

    def build_common_wavesol(self):

        # establish the wavelength solution of the first dither for all dithers
        dith1_obj = self.VP_frames[np.where(self.dith_order_lis == 1)[0][0]]
        self.wave = dith1_obj.wave

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
                hdr_comment = 'interpolated to common wavelength grid \
                (dither 1)'
                self.VP_frames[i].build_new_extension(new_ext_name, hdr_comment)

    def build_master_fiber_files(self):

        if not isinstance(self.wave, np.ndarray):
            self.build_common_wavesol()

        dith1_obj = self.VP_frames[np.where(self.dith_order_lis == 1)[0][0]]
        field_RA = dith1_obj.RA
        field_DEC = dith1_obj.DEC

        dat_lis = []
        err_lis = []
        fib_df_lis = []
        # shift fiber RA,DEC by dither file offsets
        for d in self.dith_order_lis:
            dith_obj = self.VP_frames[np.where(self.dith_order_lis == d)[0][0]]

            RA_dith_shift_as = self.dith_df[self.dith_df['dith_num'] == d]['RA_shift'].values[0]
            DEC_dith_shift_as = self.dith_df[self.dith_df['dith_num'] == d]['DEC_shift'].values[0]
            RA_dith_shift_deg = RA_dith_shift_as/3600
            DEC_dith_shift_deg = DEC_dith_shift_as/3600

            dat_lis.append(dith_obj.dat)
            err_lis.append(dith_obj.dat_err)

            dith_obj.fib_df['RA'] = field_RA+RA_dith_shift_deg
            dith_obj.fib_df['DEC'] = field_DEC+DEC_dith_shift_deg
            fib_df_lis.append(dith_obj.fib_df)

        self.master_spec = np.vstack(dat_lis)
        self.master_err_spec = np.vstack(err_lis)

        fib_df = pd.concat(fib_df_lis)
        self.master_fib_df = fib_df.reset_index(drop=True)

        # shift fiber RA,DEC by cen file offsets
        for i in range(len(self.cen_df)):
            fib_id = self.cen_df.iloc[i]['fib_id']
            RA_fib_shift_as = self.cen_df.iloc[i]['RA']
            DEC_fib_shift_as = self.cen_df.iloc[i]['DEC']
            RA_fib_shift_deg = RA_fib_shift_as/3600
            DEC_fib_shift_deg = DEC_fib_shift_as/3600

            fib_inds = self.master_fib_df[self.master_fib_df['fib_id']==fib_id].index.values

            self.master_fib_df.at[fib_inds, 'RA'] = self.master_fib_df.iloc[fib_inds]['RA'] + RA_fib_shift_deg
            self.master_fib_df.at[fib_inds, 'DEC'] = self.master_fib_df.iloc[fib_inds]['DEC'] + DEC_fib_shift_deg

    def build_data_cube(self, grid=(0, 0, 0, 0)):

        if not isinstance(self.master_spec, np.ndarray):
            self.build_master_fiber_files()

        fiberd_as = self.cen_df.iloc[0]['fiberd']  # in arcseconds
        # convert all arcsecond units to degrees
        fiberd = Angle(fiberd_as*u.arcsecond).degree

        fib_RA = self.master_fib_df['RA'].values
        fib_DEC = self.master_fib_df['DEC'].values

        regrid_size = fiberd/2.0
        kern_sig = fiberd
        max_radius = fiberd*5.0
        interp_class = interpolate_IFU.fibers_to_grid(fib_RA, fib_DEC, fiberd,
                                                      regrid_size, max_radius,
                                                      kern_sig)

        if len(set(list(grid))) == 1:
            xmin = self.master_fib_df['RA'].min()
            xmax = self.master_fib_df['RA'].max()
            ymin = self.master_fib_df['DEC'].min()
            ymax = self.master_fib_df['DEC'].max()
            grid = (xmin, xmax, ymin, ymax)

        x_grid, y_grid = interp_class.build_new_grid(grid=grid)

        self.master_fib_df['fib_flux'] = np.ones(len(self.master_fib_df))
        self.master_fib_df['fib_flux_err'] = np.zeros(len(self.master_fib_df))

        wave_frame_lis = []
        wave_frame_err_lis = []
        for i in range(len(self.wave)):
            self.master_fib_df['fib_flux'] = self.master_spec[:, i]
            self.master_fib_df['fib_flux_err'] = self.master_err_spec[:, i]

            wave_frame, wave_err_frame = interp_class.shepards_kernal()
            wave_frame_lis.append(wave_frame)
            wave_frame_err_lis.append(wave_err_frame)

        self.data_cube = np.dstack(wave_frame_lis)
        self.data_err_cube = np.dstack(wave_frame_err_lis)

    def save_data_cube(self, outname=None):
        # find mean of header values for combined dithers
        cube_hdr = self.VP_frames[0].hdr
        hdu_new = fits.PrimaryHDU(self.data_cube)
        if outname is None:
            if self.dither_group_id is not None:
                outname = self.VP_frames[0].filename.split('.')[-2]+'_data_cube_'+str(int(self.dither_group_id))+'.fits'
            else:
                dith_group_id = 1.0
                outname = self.VP_frames[0].filename.split('.')[-2]+'_data_cube_'+str(1)+'.fits'
        try:
            hdu_new.writeto(outname)
        except:
            print('Invalid save path')

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
