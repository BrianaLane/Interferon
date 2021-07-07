#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 15:28:36 2021

@author: Briana
"""
# %%
import auto_VP_run as auto_VP
import guider_observations as go
import dither_observations as do
import VP_fits_frame as vpf
import IFU_spectrum as ifu_spec
import interpolate_IFU as interp
# import emission_line_fitting_emcee
# import model_line_functions as mlf

import glob as glob

# %%
data_path = '/Volumes/B_SS/VIRUS_P/VP_reduction/20210411/redux'
guider_path = '/Volumes/B_SS/VIRUS_P/VP_reduction/20210411/guider'
dith_file = '/VP_config/dith_vp_6subdither.csv'
cen_file = '/VP_config/IFUcen_VP_new_27m.csv'

# %%
guid = go.guider_observations(guider_path)

# %%
file_list = glob.glob(data_path+'/COOLJ0931*_multi.fits')
ext = 'dithnorm'

obj_lis = []
for f in file_list:
    fits_ex = vpf.VP_fits_frame(f, ext, guide_obs=guid)
    obj_lis.append(fits_ex)


dith = do.dither_observation(obj_lis, dither_group_id=1)
# dith.normalize_dithers(guid)
dith.build_common_wavesol()
dith.build_master_fiber_files()
dith.build_data_cube()
