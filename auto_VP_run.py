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

class auto_VP_run():
    def __init__(self, data_path, fits_ext=0, guider_path=None, dith_file=None, cen_file='VP_config/IFUcen_VP_new_27m.csv', guider_as_per_pix = 0.51):
        self.data_path = data_path
        self.cen_df = pd.read_csv(cen_file, skiprows=2)
        self.orig_ext = fits_ext
        
        data_files = glob.glob(op.join(data_path, '*_*_multi.fits'))
        file_names = [f.split('/')[-1] for f in data_files]
        datetime_lis  = [dt.datetime.strptime(f.split('_')[-2],'%Y%m%dT%H%M%S') for f in file_names]
        obj_name = ['_'.join(f.split('_')[0:-2]) for f in file_names]
        
        self.data_df = pd.DataFrame({'filename':data_files, 'object':obj_name, 'dith_num':1.0, 
                                     'obs_datetime':datetime_lis, 'orig_fits_ext':self.orig_ext,
                                     'is_dither':[False]*len(data_files), 'num_obj_obs':np.NaN})
        
        if dith_file == None:
            self.is_dither_obs = False
            print('No Dither File Found: treating each file as single dither observation')
        else:
            dither_inds = [d for d in range(len(self.data_df)) if 'dither' in self.data_df.iloc[d]['filename']]
            if len(dither_inds)==0:
                print('No Dither Found: treating each file as single dither observation')
            else:
                self.dither_obs = True
                self.dith_df = pd.read_csv(dith_file, skiprows=2)
                self.data_df.at[dither_inds, 'is_dither'] = True
                
                dith_num_lis = [i.split('_')[-1] for i in self.data_df.iloc[dither_inds]['object']]
                dith_obj_lis = ['_'.join(i.split('_')[0:-2]) for i in self.data_df.iloc[dither_inds]['object']]
                self.data_df.at[dither_inds, 'dith_num'] = dith_num_lis 
                self.data_df.at[dither_inds, 'object'] = dith_obj_lis 
                
        if guider_path == None:
            self.is_guider_obs = False
        else:
            self.is_guider_obs = True
            #self.guider_obs = guider_observations(guider_path, guider_as_per_pix=guider_as_per_pix)
            
        self.object_ID_file = op.join(self.data_path, 'object_ID_list.csv')
        self.object_ID_df = None
        
        self.match_dither=False
        self.match_guider=False
    
    #find duplicate observations of the same object. mark them
    #with different indicators in data_df num_obj_obs column 
    def auto_build_dither_groups(self):
            
        if self.dither_obs:
            
            print('Grouping Dithers in Observations')
            
            obj_lis = self.data_df['object'].unique()
            obs_ct = 1
            #iterate through each unique object name in that night of data
            for o in obj_lis:
                obj_df = self.data_df[self.data_df['object']==o]
                dith_num_lis = obj_df['dith_num']
                unique_dith_names = dith_num_lis.unique()

                num_dup_lis = []
                #iterate through the list of unique dithers for that object
                #search for duplicate objects+dith_num
                for i in unique_dith_names:
                    #sort all object dithers with same dith_num by observation time
                    dith_num_inds = self.data_df[(self.data_df['object']==o) & (self.data_df['dith_num']==i)].sort_values(by='obs_datetime').index
                    #assign obs number to each dither
                    #if more than one of the same dither position it will get assigned different obs number
                    #sorting by observation time will group dithers taken at same time with same obs number
                    self.data_df.at[dith_num_inds, 'num_obj_obs'] = np.arange(len(dith_num_inds))+obs_ct
                    num_dup_lis.append(len(dith_num_inds))
                obs_ct = obs_ct+np.max(num_dup_lis)

            print('   Found', obs_ct, 'dither groups')
            
        else:
            self.data_df['num_obj_obs'] = np.arange(len(self.data_df))+1
            
        self.match_dither=True
        
    def update_data_files(self):
        #check object_ID_file for user changes in orig fits extension to be correct
        self.object_ID_df = pd.read_csv(self.object_ID_file)
        
        all_obj_lis = self.data_df['object'].unique()
        for i in all_obj_lis:
            data_obj_inds = self.data_df[self.data_df['object']==i].index
            match_ID_df = self.object_ID_df[self.object_ID_df['object']==i]
            if len(match_ID_df) > 0:
                self.data_df.at[data_obj_inds, 'orig_fits_ext'] = int(match_ID_df['fits_ext'])
                self.data_df.at[data_obj_inds, 'if_flux_stand'] = int(match_ID_df['is_flux_stand'])
            else:
                self.data_df.drop(data_obj_inds, inplace=True)
                self.data_df.reset_index(drop=True, inplace=True)
                
    def normalize_all_dithers(self):
        return None