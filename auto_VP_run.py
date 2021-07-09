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


class VP_run():

    def __init__(self, data_path, fits_ext=0, fits_err_ext=3, guider_path=None, 
                 dith_file=None, cen_file='VP_config/IFUcen_VP_new_27m.csv',
                 guider_as_per_pix=0.51):

        self.data_path = data_path
        self.dith_file = dith_file
        self.cen_file = cen_file

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

        self.data_df = pd.DataFrame({'filename': data_files,
                                     'object': obj_name, 'dith_num': 1.0,
                                     'obs_datetime': datetime_lis,
                                     'orig_fits_ext': self.orig_ext,
                                     'is_dither': [False]*len(data_files),
                                     'dither_group_id': np.NaN})

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

        # self.object_ID_file = op.join(self.data_path, 'object_ID_list.csv')
        # self.object_ID_df = None

        self.match_dither = False

    # find duplicate observations of the same object. mark them
    # with different indicators in data_df num_obj_obs column
    def build_dither_groups(self):

        if self.dither_obs:

            print('Grouping Dithers in Observations')

            obj_lis = self.data_df['object'].unique()
            obs_ct = 1
            # iterate through each unique object name in that night of data
            for o in obj_lis:
                obj_df = self.data_df[self.data_df['object'] == o]
                dith_num_lis = obj_df['dith_num']
                unique_dith_names = dith_num_lis.unique()

                num_dup_lis = []
                # iterate through the list of unique dithers for that object
                # search for duplicate objects+dith_num
                for i in unique_dith_names:
                    # sort all object dithers with same dith_num by
                    # observation time
                    dith_num_inds = self.data_df[(self.data_df['object']==o) & (self.data_df['dith_num']==i)].sort_values(by='obs_datetime').index
                    # assign obs number to each dither
                    # if more than one of the same dither position it will get
                    # assigned different obs number
                    # sorting by observation time will group dithers taken at
                    # same time with same obs number
                    self.data_df.at[dith_num_inds, 'dither_group_id'] = np.arange(len(dith_num_inds))+obs_ct
                    num_dup_lis.append(len(dith_num_inds))
                obs_ct = obs_ct+np.max(num_dup_lis)

            print('   Found', obs_ct, 'dither groups')

        else:
            self.data_df['dither_group_id'] = np.arange(len(self.data_df))+1

        self.match_dither = True

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
            guid = go.guider_observations(self.guider_path,
                                          guider_as_per_pix=self.guider_as_per_pix)
            self.guider_obs = guid
        else:
            print('NO GUIDER PATH PROVIDED')

    def dither_object(self, dith_grp_id, norm=True):

        if not self.match_dither:
            self.build_dither_groups()

        if not isinstance(self.guider_obs, go.guider_observations):
            self.obs_guider()

        dith_grp_df = self.data_df[self.data_df['dither_group_id'] == dith_grp_id]
        dith_obj_lis = []
        for i in range(len(dith_grp_df)):
            VP_file = dith_grp_df.iloc[i].filename
            fits_ex = vpf.VP_fits_frame(VP_file, self.fits_ext,
                                        fits_err_ext=self.fits_err_ext,
                                        cen_file=self.cen_file,
                                        guide_obs=self.guider_obs)

            dith_obj_lis.append(fits_ex)
        dith = do.dither_observation(dith_obj_lis, dither_group_id=dith_grp_id,
                                     dith_file=self.dith_file)

        if norm:
            dith.normalize_dithers(self.guider_obs)

        return dith

    def build_data_cube(self, dith):
        dith.write_data_cube()

    def run_all_dithers(self, norm=True):

        if not self.match_dither:
            self.build_dither_groups()

        dith_group_lis = self.data_df['dither_group_id'].unique()
        print('AUTO-BUILD data cubes for ', len(dith_group_lis), ' dither groups')
        for g in dith_group_lis:
            dith = self.dither_object(g, norm=norm)
            self.build_data_cube(dith)
        if norm:
            self.fits_ext = dith.VP_frames[0].fits_ext
