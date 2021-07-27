#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 15:28:36 2021

@author: Briana
"""
import argparse as ap
from auto_VP_run import VP_run

default_dith_file = 'VP_config/dith_vp_6subdither.csv'
default_cen_file = 'VP_config/IFUcen_VP_new_27m.csv'

parser = ap.ArgumentParser(add_help=True)

parser.add_argument("data_folder",
                    help='''Folder containing Remedy mulit.fits files''',
                    type=str)

parser.add_argument("guider_folder",
                    help='''Folder containing guider .fits files''',
                    type=str)

parser.add_argument('-d', '--dith_file', type=str,
                    help='''Name of dither file (DEFAULT:dith_vp_6subdither.csv)''',
                    default=default_dith_file)

parser.add_argument('-c', '--cen_file', type=str,
                    help='''Name of cen file (DEFAULT:IFUcen_VP_new_27m.csv)''',
                    default=default_cen_file)

parser.add_argument('-g', '--guidecam_plate_scale', type=float,
                    help='''arcseconds per pixel for guider camera''',
                    default=0.51)

args = parser.parse_args(args=None)

vp1 = VP_run(args.data_folder, fits_ext=0,
             guider_path=args.guider_folder,
             dith_file=args.dith_file,
             cen_file=args.cen_file,
             guider_as_per_pix=args.guidecam_plate_scale)


vp1.run_all_dithers(norm=True)


