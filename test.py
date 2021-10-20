#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 15:12:59 2021

@author: Briana
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 15:28:36 2021

@author: Briana
"""
# %%
from auto_VP_run import VP_run

data_path1 = '/Volumes/BRI_DRIVE/VIRUSP/COOL_Lamps_data/20210410/redux_obj_grps'
guider_path1 = '/Volumes/BRI_DRIVE/VIRUSP/COOL_Lamps_data/20210410/guider'
dith_file = 'VP_config/dith_vp_6subdither.csv'
cen_file = 'VP_config/IFUcen_VP_new_27m.csv'

vpt = VP_run(data_path1, fits_ext=0, groupby_obj=True,
                             guider_path=guider_path1, dith_file=dith_file,
                             cen_file=cen_file, guider_as_per_pix=0.51)

# %%
vpt.run_all_groups(norm=True, correct_airmass=False, flux_calib=False)

# %%
from data_cube import Cube

sta_cu = Cube(data_path1+'/feige66_dither_1_20210410T024048_multi_data_cube_3.fits')
#obj_cu = Cube(data_path1+'/COOLJ0931_dither_1_20210411T030846_multi_data_cube_1.fits')

# %%
sta_cu.build_sensitiviy_curve(cal_spec=None, cal_wave=None,
                              cal_err=None, cal_per_err=5.0,
                              save=True, plot=True)

# %%
sens_curve_file = data_path1+'/BD_33D2642_dither_1_20210411T082744_multi_data_cube_2_SENS_CURV.csv'
obj_cu.flux_calibrate(sens_curve_file, save=True)

# %%
from auto_VP_run import VP_run

date = '20210411'
data_path = '/Volumes/BRI_DRIVE/VIRUSP/COOL_Lamps_data/'+date+'/redux'
guider_path = '/Volumes/BRI_DRIVE/VIRUSP/COOL_Lamps_data/'+date+'/guider'

#data_path = '/Volumes/BRI_DRIVE/VIRUSP/TAURUS_2021_data/'+date+'/redux'
#guider_path = '/Volumes/BRI_DRIVE/VIRUSP/TAURUS_2021_data/'+date+'/guider'
dith_file = 'VP_config/dith_vp_6subdither.csv'
cen_file = 'VP_config/IFUcen_VP_new_27m.csv'

# %%
vp1 = VP_run(data_path, fits_ext=0,
                             guider_path=guider_path, dith_file=dith_file,
                             cen_file=cen_file, guider_as_per_pix=0.51)

# %%
vp1.run_all_groups(norm=True, correct_airmass=False, flux_calib=False)

# %%
obj_df = vp1.build_obj_df(obj_name=None, dither_group_id=1)
dith = vp1.dither_object(obj_df, norm=True)
#vp1.build_data_cube(dith)

# %%

# import auto_VP_run as auto_VP
# import IFU_spectrum as ifu_spec
# import emission_line_fitting_emcee
# import model_line_functions as mlf

import glob as glob
import guider_observations as go
import dither_observations as do
import VP_fits_frame as vpf

guid = go.guider_obs(guider_path)

# %%

file_list = glob.glob(data_path + '/COOLJ1012+0558*_multi.fits')
print(len(file_list))
fits_ex = vpf.VP_frame(file_list[0], fits_ext=0, guide_obs=guid,
                            run_correct_airmass=True)

# %%

file_list = glob.glob(data_path + '/COOLJ1012+0558*_multi.fits')
# ext = 'dithnorm'
ext = 0

obj_lis = []
for f in file_list:
    fits_ex = vpf.VP_frame(f, ext, guide_obs=guid,
                                run_correct_airmass=True)
    obj_lis.append(fits_ex)

dith = do.dither_observation(obj_lis, dither_group_id=1)

# %%
dith.normalize_dithers(guid)
dith.build_common_wavesol()
dith.build_master_fiber_files()

# %%

dith.write_data_cube()
# %%

from data_cube import Cube

date = '20210411'
data_path = '/Volumes/B_SS/VIRUS_P/VP_reduction/COOL_Lamps_data/'+date+'/redux'
cube_file = data_path+'/BD_33D2642_dither_1_20210411T082744_multi_data_cube_1.fits'
err_cube_file = data_path+'/BD_33D2642_dither_1_20210411T082744_multi_data_cube_1_err.fits'

cu = Cube(cube_file, err_cube_file=err_cube_file)
