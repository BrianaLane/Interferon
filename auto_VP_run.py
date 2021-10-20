#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 15:28:36 2021

@author: Briana
"""

import os.path as op
import sys
import pandas as pd
import numpy as np
import glob
import datetime as dt

import guider_observations as go
import dither_observations as do
import VP_fits_frame as vpf
import image_utils


class VP_run():

    def __init__(self, data_path, fits_ext=0, fits_err_ext=3,
                 guider_path=None, dith_file=None,
                 groupby_obj=True, flux_calib=True, grating='VP1',
                 cen_file='VP_config/IFUcen_VP_new_27m.csv',
                 guider_as_per_pix=0.51):

        self.data_path = data_path
        self.dith_file = dith_file
        self.cen_file = cen_file
        self.grating = grating

        self.orig_ext = fits_ext
        self.fits_ext = fits_ext
        self.orig_err_ext = fits_err_ext
        self.fits_err_ext = fits_err_ext

        data_files = glob.glob(op.join(data_path, '*_*_multi.fits'))
        if len(data_files)==0:
            sys.exit('No data files found: check path')
        file_names = [f.split('/')[-1] for f in data_files]
        datetime_lis  = [dt.datetime.strptime(f.split('_')[-2],'%Y%m%dT%H%M%S') for f in file_names]
        obj_name = ['_'.join(f.split('_')[0:-2]) for f in file_names]

        self.data_df = pd.DataFrame({'object': obj_name,
                                     'dith_num': 1.0,
                                     'obs_datetime': datetime_lis,
                                     'airmass': np.NaN,
                                     'dither_group_id': np.NaN,
                                     'stand_match_name': None,
                                     'stand_match_dith_id': np.NaN,
                                     'is_stand': [False]*len(data_files),
                                     'is_dither': [False]*len(data_files),
                                     'orig_fits_ext': self.orig_ext,
                                     'filename': data_files})

        self.dither_obs = False
        if self.dith_file is None:
            print('No Dither File Found: treating each file as single dither observation')
        else:
            dither_inds = [d for d in range(len(self.data_df)) if 'dither' in self.data_df.iloc[d]['filename']]
            if len(dither_inds) == 0:
                print('No Dither Found: treating each file as single dither observation')
            else:
                self.dither_obs = True
                self.data_df.at[dither_inds, 'is_dither'] = True

                dith_num_lis = [i.split('_')[-1] for i in self.data_df.iloc[dither_inds]['object']]
                dith_obj_lis = ['_'.join(i.split('_')[0:-2]) for i in self.data_df.iloc[dither_inds]['object']]
                self.data_df.at[dither_inds, 'dith_num'] = dith_num_lis
                self.data_df.at[dither_inds, 'object'] = dith_obj_lis

        self.guider_path = guider_path
        self.guider_obs = None
        self.guider_as_per_pix = guider_as_per_pix

        self.obj_lis = self.data_df['object'].unique()
        self.sci_obj_lis = []
        self.stand_obj_lis = []

        self.object_ID_file = op.join(self.data_path, 'object_ID_list.csv')

        self.build_dither_groups(groupby_obj=groupby_obj)
        self.auto_find_standards()

    # find duplicate observations of the same object. mark them
    # with different indicators in data_df num_obj_obs column
    def build_dither_groups(self, groupby_obj=True):

        if groupby_obj:
            print('Grouping Objects in Observations')

            for i in range(len(self.obj_lis)):
                obj_inds = self.data_df[self.data_df['object'] == self.obj_lis[i]].index
                self.data_df.at[obj_inds, 'dither_group_id'] = i+1

            print('   Found', len(set(self.data_df['dither_group_id'].values)),
                  'object groups')

        else:

            if self.dither_obs:
                print('Grouping Dithers in Observations')

                obs_ct = 1
                # iterate through each unique object name in that night of data
                for o in self.obj_lis:
                    obj_df = self.data_df[self.data_df['object'] == o].sort_values(by='obs_datetime')
                    
                    dith_num_lis = []
                    df_ct = 0
                    for i in range(len(obj_df)):
                        dith_ind = obj_df.index[i]
                        dith_num = int(obj_df.iloc[i]['dith_num'])
                        dith_num_lis.append(dith_num)
                        df_ct = df_ct+1
                        if len(set(dith_num_lis)) != df_ct:
                            obs_ct = obs_ct+1
                            dith_num_lis = [dith_num]
                            df_ct = 1
                        self.data_df.at[dith_ind, 'dither_group_id'] = obs_ct
                    obs_ct = obs_ct+1

                print('   Found', len(set(self.data_df['dither_group_id'].values)), 'dither groups')

            else:
                print('No Groupings Found: treating each image separately')
                self.data_df['dither_group_id'] = np.arange(len(self.data_df))+1
                
    def auto_find_standards(self):
        # check each object to see if name is match for a standard star name
        # if it is then auto determine it is a standard observation
        known_stand_df = pd.read_csv('standard_stars/standard_stars.csv')
        star_names = list(known_stand_df['star_ID'].values)
        fits_names = list(known_stand_df['file_ID'].values)
        alt_names1 = list(known_stand_df['simbad_ID'].values)
        alt_names2 = list(known_stand_df['alt_simbad_ID'].values)

        alt_lis = [fits_names, alt_names1, alt_names2]

        for o in self.obj_lis:
            obj_inds = self.data_df[self.data_df['object']==o].index
            best_match, score = image_utils.match_obj_to_lis(o, star_names)

            # if score isn't almost perfect check through alt name list for
            # better name match
            if score < 0.95:

                for l in alt_lis:
                    new_match, new_score = image_utils.match_obj_to_lis(o, l)

                    if new_score > score:
                        score = new_score
                        best_match = new_match
                        
            # if the final score is decent then call this object a standard
            if score > 0.75:
                print('   Classifying', o, 'as a standard star')
                self.data_df.at[obj_inds, 'is_stand'] = True
            
    def auto_pair_standards(self, pair_by='datetime'):
        # for objects that are not standards pair them with closest standard
        # if pair_by = 'datetime': pairing based on timing of exposure
        # if pair_by = 'airmass': pairing based on most similar airmass
        
        #stand_ids = self.data_df[self.data_df['is_stand']==True]['dither_group_id'].unique()
        #sci_ids = self.data_df[self.data_df['is_stand']==False]['dither_group_id'].unique()
        
        stand_df = self.data_df[self.data_df['is_stand']==True]
        
        if len(stand_df) < 1:
            raise ValueError('No standards found in observation')

        else:
            if pair_by == 'datetime':
                
                stand_dict = {}
                sci_dict = {}
                
                for i in self.data_df['dither_group_id'].unique:
                    dith_df = self.data_df[self.data_df['dither_group_id']==i]
                    s = pd.Series(dith_df['obs_datetime'])
                    
                    if dith_df.iloc[0]['is_stand']:
                        stand_dict[i] = s.mean()
                        
                    else:
                        sci_dict[i] = s.mean()
                        
                
                        
                    # now need to pair the sci objs with standards from dict
                
            elif pair_by == 'airmass':
                for i in range(len(self.data_df)):
                    filename = self.data_df.iloc[0].filename
                    
                    # need to find airmass of each source
                    # pair them 
                
            else:
                raise ValueError('pair_by must either be datetime or airmass')
            

    def update_data_files(self):
        # check object_ID_file for user changes in orig fits
        # extension to be correct
        self.object_ID_df = pd.read_csv(self.object_ID_file)

        all_obj_lis = self.data_df['object'].unique()
        for i in all_obj_lis:
            data_obj_inds = self.data_df[self.data_df['object'] == i].index
            match_ID_df = self.object_ID_df[self.object_ID_df['object'] == i]

            if len(match_ID_df) > 0:
                self.data_df.at[data_obj_inds, 'orig_fits_ext'] = int(match_ID_df['fits_ext'])
                self.data_df.at[data_obj_inds, 'if_flux_stand'] = int(match_ID_df['is_flux_stand'])

            else:
                self.data_df.drop(data_obj_inds, inplace=True)
                self.data_df.reset_index(drop=True, inplace=True)
     
    def obs_guider(self):
        if self.guider_path is not None:
            guid = go.guider_obs(self.guider_path, guider_as_per_pix=self.guider_as_per_pix)
            self.guider_obs = guid

        else:
            print('NO GUIDER PATH PROVIDED')

    def dither_object(self, dith_grp_df, norm=True, correct_airmass=False):

        if not isinstance(self.guider_obs, go.guider_obs):
            self.obs_guider()

        dith_grp_id = dith_grp_df['dither_group_id'].values[0]
        dith_obj_lis = []
        for i in range(len(dith_grp_df)):
            VP_file = dith_grp_df.iloc[i].filename
            fits_ex = vpf.VP_frame(VP_file, self.fits_ext,
                                        fits_err_ext=self.fits_err_ext,
                                        grating=self.grating,
                                        cen_file=self.cen_file,
                                        guide_obs=self.guider_obs,
                                        run_correct_airmass=correct_airmass)

            dith_obj_lis.append(fits_ex)
        dith = do.dither_observation(dith_obj_lis, dither_group_id=dith_grp_id,
                                     dith_file=self.dith_file)

        if norm:
            dith.normalize_dithers(self.guider_obs)

        return dith

    def build_data_cube(self, dith):
        dith.write_data_cube()
        
    def build_obj_df(self, obj_name=None, dither_group_id=None):

        if isinstance(obj_name, str):
            obj_df = self.data_df[self.data_df['object'] == obj_name]

            if len(obj_df) > 0:

                if isinstance(dither_group_id, float, int, np.int64):
                    obj_df = self.obj_df[self.obj_df['dither_group_id']==dither_group_id]

                    if len(obj_df) > 0:
                        # valid obj name and dith ID
                        return obj_df

                    else:
                        raise ValueError('Can not find ' + str(dither_group_id) + ' for object '+obj_name)

                else:
                    # only valid obj_name
                    if set(self.data_df['dither_group_id']) > 1:
                        new_dith_id = np.amax(self.data_df['dither_group_id']) + 1
                        obj_df.at['dither_group_id'] == new_dith_id
                    return obj_df

            else:
                raise ValueError('Object name '+obj_name+' not found')

        elif isinstance(dither_group_id, (float, int, np.int64)):
            obj_df = self.data_df[self.data_df['dither_group_id']==dither_group_id]

            if len(obj_df) > 0:
                # only valid dith ID
                return obj_df

        else:
            # neither valid dith ID or object name provided
            raise ValueError('Need to specify either obj_name or dither_group_id')

    def run_object(self, obj_name=None, obj_dith_id=None, 
                   norm=True, correct_airmass=False, flux_calib=True):

        obj_df = self.build_obj_df(obj_name=obj_name,
                                   dither_group_id=obj_dith_id)

        obj_name = obj_df['object'].values[0]
        obj_dith_id = obj_df['dither_group_id'].values[0]
        print('AUTO-BUILD data cube for ', str(obj_name), ' with dither group '+str(obj_dith_id))

        dith = self.dither_object(obj_df, norm=norm,
                                  correct_airmass=correct_airmass)
        
        self.build_data_cube(dith)

    def run_all_groups(self, norm=True, correct_airmass=False,
                       flux_calib=True):

        dith_group_lis = self.data_df['dither_group_id'].unique()
        print('AUTO-BUILD data cubes for ', len(dith_group_lis), ' dither groups')

        for g in dith_group_lis:
            dith_grp_df = self.data_df[self.data_df['dither_group_id']==g]
            dith = self.dither_object(dith_grp_df, norm=norm,
                                      correct_airmass=correct_airmass)
            self.build_data_cube(dith)

        # updates the fits extentsion for the auto_VP_run class
        # but only set update to this of all groups have been reduced 
        self.fits_ext = dith.VP_frames[0].fits_ext
            
        #if flux_calib:
            
