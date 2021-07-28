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

date = '20210710'
#data_path = '/Volumes/B_SS/VIRUS_P/VP_reduction/COOL_Lamps_data/'+date+'/redux_2'
#guider_path = '/Volumes/B_SS/VIRUS_P/VP_reduction/COOL_Lamps_data/'+date+'/guider'

data_path = '/Volumes/B_SS/VIRUS_P/VP_reduction/TAURUS_2021_data/'+date+'/redux'
guider_path = '/Volumes/B_SS/VIRUS_P/VP_reduction/TAURUS_2021_data/'+date+'/guider'
dith_file = 'VP_config/dith_vp_6subdither.csv'
cen_file = 'VP_config/IFUcen_VP_new_27m.csv'

# %%
vp1 = VP_run(data_path, fits_ext=0,
                             guider_path=guider_path, dith_file=dith_file,
                             cen_file=cen_file, guider_as_per_pix=0.51)

vp1.build_dither_groups()

# %%
vp1.run_all_dithers(norm=True)

# %%
dith = vp1.dither_object(1, norm=True)
vp1.build_data_cube(dith)

# %%
vp1.build_data_cube(dith)

# %%

# import auto_VP_run as auto_VP
# import IFU_spectrum as ifu_spec
# import emission_line_fitting_emcee
# import model_line_functions as mlf

import glob as glob
import guider_observations as go
import dither_observations as do
import VP_fits_frame as vpf

#guid = go.guider_observations(guider_path)

# %%

file_list = glob.glob(data_path + '/COOLJ0931*_multi.fits')
ext = 'dithnorm'
#ext = 0

obj_lis = []
for f in file_list:
    fits_ex = vpf.VP_fits_frame(f, ext, guide_obs=None)
    obj_lis.append(fits_ex)


dith = do.dither_observation(obj_lis, dither_group_id=1)
# %%
#dith.normalize_dithers(guid)
#dith.build_common_wavesol()
#dith.build_master_fiber_files()
dith.write_data_cube()

# %%

from data_cube import Cube

date = '20210411'
data_path = '/Volumes/B_SS/VIRUS_P/VP_reduction/COOL_Lamps_data/'+date+'/redux'
cube_file = data_path+'/BD_33D2642_dither_1_20210411T082744_multi_data_cube_1.fits'
err_cube_file = data_path+'/BD_33D2642_dither_1_20210411T082744_multi_data_cube_1_err.fits'

cu = Cube(cube_file, err_cube_file=err_cube_file)
