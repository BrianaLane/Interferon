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
import guider_observations as go
import IFU_spectrum as ifu_spec
import VP_fits_frame as vpf

class dither_observation():
    def __init__(self, VP_frames, dither_group_id=None, dith_file='VP_config/dith_vp_6subdither.csv'):
        
        self.VP_frames = VP_frames

        self.dith_order_lis = np.ones(len(VP_frames))
        for f in range(len(self.VP_frames)):
            if not isinstance(self.VP_frames[f], vpf.VP_fits_frame):
                print('Must provide list of vpf.VP_fits_frame objects for dither set')
                return None
            else:
                self.dith_order_lis[f] = self.VP_frames[f].dith_num
                if dither_group_id != None:
                    self.VP_frames[f].dither_group_id = dither_group_id
        
        self.dither_group_id = dither_group_id
        self.dith_df = pd.read_csv(dith_file, skiprows=2)
        
        self.wave = None
        self.master_spec = None
        self.master_err_spec = None
        self.master_fib_df = None
        
        self.data_cube = None
        self.data_err_cube = None

                
    def normalize_dithers(self, guide_obs, star_thres=10., num_bright_stars=10, star_fwhm=8.0, fwhm_lim=(0.5,10), mag_lim=10):
        if isinstance(guide_obs, go.guider_observations):
        
            #check if matched each fits image has matched guider frames
            obs_guide_lis = []
            for f in range(len(self.VP_frames)):
                if self.VP_frames[f].guide_match == None:
                    print('MATCHING GUDIE FRAMES')
                    self.VP_frames[f].match_guider_frames(guide_obs)
                obs_guide_lis.append(self.VP_frames[f].guider_ind)
            obs_guide_lis = np.hstack(obs_guide_lis)
                
            #find the RA/DEC dither shift in pixels in guider cam frames
            self.dith_df['RA_pix_shift'] = self.dith_df['RA_shift']/guide_obs.guider_ps
            self.dith_df['DEC_pix_shift'] = self.dith_df['DEC_shift']/guide_obs.guider_ps
            
            #find a guider image to use as reference for stars
            ref_sources_df, ref_guide_ind = guide_obs.find_ref_guide_frame(obs_guide_lis, star_thres=star_thres, 
                                                    num_bright_stars=num_bright_stars, star_fwhm=star_fwhm, 
                                                    fwhm_lim=fwhm_lim, mag_lim=mag_lim)
            
            guide_stars_lis = []
            for f in self.VP_frames:
                guide_ind_lis = f.guider_ind
                frame_dith_num = f.dith_num
                
                #update ref_sources_df with the correct dither coordinates
                shift_df = self.dith_df[self.dith_df['dith_num']==frame_dith_num]
                x_mod = shift_df['RA_pix_shift'].values[0]
                y_mod = shift_df['DEC_pix_shift'].values[0]
                mod_ref_sources_df = ref_sources_df.copy()
                mod_ref_sources_df['xcentroid'] = ref_sources_df['xcentroid']-x_mod
                mod_ref_sources_df['ycentroid'] = ref_sources_df['ycentroid']-y_mod
        
                for g in guide_ind_lis:
                    sources_fit = guide_obs.measure_guide_star_params(g, mod_ref_sources_df)
                    sources_fit = guide_obs.flag_stars(sources_fit.copy(), fwhm_lim=fwhm_lim, mag_lim=mag_lim)
                    sources_fit['dith_num'] = frame_dith_num
                    sources_fit['guide_ind'] = g
                    
                    good_sources_fit = sources_fit[sources_fit['bad_flag']==False]
                    guide_stars_lis.append(good_sources_fit)
                    
            guide_stars_df = pd.concat(guide_stars_lis).reset_index()
            dith_stars_df = guide_stars_df.groupby(by=['dith_num','id'], as_index=False).mean()

            dith_star_count = dith_stars_df[['id', 'dith_num']].groupby(by='dith_num', as_index=False).count()
            dith_star_count.rename(columns={'id':'num_stars_used'}, inplace=True)

            dith_flux_sum = dith_stars_df[['dith_num', 'flux_fit']].groupby(by='dith_num', as_index=False).sum()
            dith_flux_sum['flux_norm'] = dith_flux_sum['flux_fit']/dith_flux_sum['flux_fit'].max()

            dith_fwhm_avg = dith_stars_df[['dith_num', 'fwhm(arcseconds)']].groupby(by='dith_num', as_index=False).mean()

            dith_norm_df1 = dith_flux_sum.merge(dith_fwhm_avg, on='dith_num', how='outer')
            dith_norm_df = dith_norm_df1.merge(dith_star_count, on='dith_num', how='outer')

            for i in self.VP_frames: 
                frame_dith_num = i.dith_num
                see_val = dith_norm_df[dith_norm_df['dith_num']==frame_dith_num]['fwhm(arcseconds)'].values[0]
                norm_val = dith_norm_df[dith_norm_df['dith_num']==frame_dith_num]['flux_norm'].values[0]
                
                i.seeing = see_val
                i.dith_norm = norm_val
                dith_norm_dat = i.dat*norm_val
                i.dat = dith_norm_dat
                    
                i.build_new_extension('dithnorm', 'normalize dither spec from guide star flux')
                
        else:
            print('guide_obs must be a guider_observations class object')
            return None

                    
    def build_common_wavesol(self):
        
        #establish the wavelength solution of the first dither for all dithers
        dith1_obj = self.VP_frames[np.where(self.dith_order_lis == 1)[0][0]]
        self.wave = dith1_obj.wave

        for i in range(len(self.VP_frames)):
            frame_wave = self.VP_frames[i].wave
            
            if not np.array_equal(frame_wave, self.wave):
                print('Fixing dither', self.VP_frames[i].dith_num, 'to common wavelength grid')
                self.VP_frames[i].wave = self.wave
                old_dat = self.VP_frames[i].dat
                for i in range(np.shape(old_dat)[0]):
                    spec_obj = ifu_spec.IFU_spectrum(old_dat[i], frame_wave)
                    spec_obj.new_wave_sol(self.wave)
                    new_spec_lis.append(spec_obj.spec)
                new_dat = np.vstack(new_spec_lis)
                new_ext_name = 'comwave'
                hdr_comment = 'interpolated to common wavelength grid (dither 1)'
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
        for d in self.dith_order_lis :
            dith_obj = self.VP_frames[np.where(self.dith_order_lis == d)[0][0]]
            RA_shift = self.dith_df[self.dith_df['dith_num']==d]['RA_shift'].values[0]
            DEC_shift = self.dith_df[self.dith_df['dith_num']==d]['DEC_shift'].values[0]
            
            dat_lis.append(dith_obj.dat)
            err_lis.append(dith_obj.dat_err)
            fib_df_lis.append(dith_obj.fib_df)
            dith_obj.fib_df['RA'] = dith_obj.fib_df['RA']+RA_shift
            dith_obj.fib_df['DEC'] = dith_obj.fib_df['RA']+DEC_shift
            
        self.master_spec = np.vstack(dat_lis)
        self.master_err_spec = np.vstack(err_lis)
        self.master_fib_df = pd.concat(fib_df_lis)

    def interpolate_frame(self, flux_col='fib_flux', err_col='fib_flux_err'):
        
        if not isinstance(self.master_spec, np.ndarray):
            self.build_master_fiber_files()
        
        fiberd_as = self.cen_df.iloc[0]['Fiber_d'] # in arcseconds
        #convert all arcsecond units to degrees 
        fiberd = Angle(fiberd_as*u.arcsecond).degree

        regrid_size = fiberd/2.0
        kern_sig = fiberd
        max_radius = fiberd*5.0
        interp_class = interpolate_IFU.fibers_to_grid(fib_df[flux_col], fib_df[err_col], 
                                                      fib_df['RA'], fib_df['DEC'], fiberd, 
                                                      regrid_size, max_radius, kern_sig)
        
        x_grid, y_grid = interp_class.build_new_grid()
        flux_grid, error_grid = interp_class.shepards_kernal()
        
        return flux_grid, error_grid
    
    def build_data_cube(self):
        
        wave_frame_lis = []
        wave_frame_err_lis = []
        for i in range(len(self.wave)):
            fib_df = self.master_fib_df.copy()
            fib_df['fib_flux'] = self.master_spec[::i]
            fib_df['fib_err_flux'] = self.master_err_spec[::i]
            wave_frame, wave_err_frame = interpolate_frame(fib_df)
            wave_frame_lis.append(wave_frame)
            wave_frame_err_lis.append(wave_err_frame)
            
        self.data_cube = wave_frame_lis.dstack(wave_frame_lis)
        self.data_err_cube = wave_frame_lis.dstack(wave_frame_err_lis)
        
    def fit_emission_lines(self, line_model):
        if not isinstance(self.master_spec, np.ndarray):
            self.build_master_fiber_files()
        
        for i in range(len(master_fib_df)):
            fib = master_fib_df.iloc[i]
            spec = fib[self.master_spec[fib.index]]
            err_spec = fib[self.master_err_spec[fib.index]]
        
        self.master_fib_df[line_model+'_flux'] = flux
        self.master_fib_df[line_model+'_err'] = flux_err
        
    def build_emission_line_image(self, line_model):
        self.fit_emission_lines()
        flux_grid, error_grid = self.interpolate_frame(flux_col=line_model+'_flux', err_col=line_model+'_err')
        
        return flux_grid, error_grid
        
    def get_throughput(self):
        return None

    def flux_calib(self):
        return None 
        
    #fib_ind (list/array)(optional, default will sum all fibers): list of fiber indices to sum and plot
    def build_frame_sum_spec(self, fib_inds=[], z=np.NaN, plot=False):
        if len(fib_inds) == 0:
            spec = np.sum(self.dat, axis=0)
        elif len(fib_inds)==1:
            spec = self.dat[fib_inds[0]]
        else:
            spec = np.sum(self.dat[fib_inds,:], axis=0)
        
        sum_spec = ifu_spec.IFU_spectrum(spec, self.wave, z=z)
        
        if plot:
            sum_spec.plot_spec(spec_units='Electrons per second')
            
        return sum_spec
                
            