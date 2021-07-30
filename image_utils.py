#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 16:13:46 2021

@author: Briana
"""

import sys
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import pandas as pd

from astropy.io import fits, ascii
from astropy.wcs import WCS
import astropy.visualization as av
from astropy.modeling import models, fitting
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
from astropy import coordinates as coords
from astropy.stats import sigma_clipped_stats
from astropy.utils.exceptions import AstropyWarning
import warnings

with warnings.catch_warnings():
   warnings.filterwarnings("ignore")
   from photutils import DAOStarFinder

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from astroquery.sdss import SDSS
from photutils import CircularAperture, aperture_photometry, CircularAnnulus


def plot_subframe(ax, frame, vmin=None, vmax=None, c_map='Greys'):

    norm = av.ImageNormalize(frame, interval=av.ZScaleInterval(),
                             stretch=av.SqrtStretch())

    if (vmin is None) or (vmax is None):
        ax.imshow(frame, cmap=c_map, origin='lower', norm=norm)
    else:
        ax.imshow(frame, cmap=c_map, origin='lower', norm=norm,
                  vmin=vmin, vmax=vmax)

    ax.set_xlabel('RA', fontsize=25)
    ax.set_ylabel('DEC', fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=15)


def plot_frame(frame, wcs=None, vmin=None, vmax=None, c_map='Greys',
               save=False, outfile=None):
    
    if wcs.naxis==3:
        wcs = WCS(wcs.to_header(), naxis=2)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=wcs)

    plot_subframe(ax, frame, vmin=vmin, vmax=vmax, c_map=c_map)

    if save:
        try:
            plt.savefig(outfile)
        except:
            print('Invaild outfile path to save plot')

    plt.show()


def find_stars(frame, star_thres=10., num_bright_stars=10,
               star_fwhm=8.0, plot_sources=False):

    mean, median, std = sigma_clipped_stats(frame, sigma=3.0)
    daofind = DAOStarFinder(fwhm=star_fwhm, threshold=star_thres*std,
                            peakmax=64000, exclude_border=True,
                            brightest=num_bright_stars)

    sources = daofind(frame)
    
    if sources is None:
        return None

    else:
        for col in sources.colnames:
            sources[col].info.format = '%.8g'  # for consistent table output
        sources_df = sources.to_pandas()
    
        if plot_sources:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            plot_subframe(ax, frame)
            for i in range(len(sources_df)):
                aperture = CircularAperture((sources_df.iloc[i]['xcentroid'],
                                             sources_df.iloc[i]['ycentroid']),
                                            r=star_fwhm)
    
                aperture.plot(color='red', lw=2.5)
            plt.show()
    
        return sources_df.copy()


def get_star_mag(frame, wcs):

    g_coords = coords.SkyCoord(hdr['RA'], hdr['DEC'])
    params = {'nDetections.min': 0, 'gQfPerfect.min': 0.85,
             'rQfPerfect.min': 0.85,'iQfPerfect.min': 0.85,
             'gMeanPSFMag.min': 0, 'rMeanPSFMag.min': 0, 'iMeanPSFMag.min': 0}

    return sources_df


def measure_star_params(frame, sources_df, plot_star_cutouts=False):

    for s in range(len(sources_df)):
        so = sources_df.iloc[s]
        xp, yp = (int(so['xcentroid']), int(so['ycentroid']))
        # this defines half width/height of square cutout
        off = 10

        # makes sure it is not too close to the edge to create cutout
        if (xp > (off-1)) and (yp > (off-1)):
            # build cutout image
            im = frame[yp-off:yp+off, xp-off:xp+off]
            # create array of x and y coordinates of cutout for the fitter
            y_grid, x_grid = np.mgrid[:np.shape(im)[0], :np.shape(im)[1]]
            # finds center for new coodinates of cutout
            x_cen, y_cen = (xp - (xp-off), yp - (yp-off))

            g = models.Gaussian2D(amplitude=so['peak'], x_mean=x_cen,
                                  y_mean=y_cen, x_stddev=1.0, y_stddev=1.0)

            gfit = fitting.LevMarLSQFitter()

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', AstropyWarning)
                # fit 2D Guassian model to cutout image
                fit_g = gfit(g, x_grid, y_grid, im)

            sd = np.average([fit_g.x_stddev.value, fit_g.y_stddev.value])
            xcen_fit_s, ycen_fit_s = (fit_g.x_mean.value, fit_g.y_mean.value)
            fwhm_fit = 2.*np.sqrt(2.*np.log(2))*sd
            flux_fit = 2*np.pi*fit_g.amplitude.value*fit_g.x_stddev.value*fit_g.y_stddev.value
            mag_fit = -2.5*np.log10(flux_fit)

            sources_df.at[s, 'xcentroid_fit'] = xcen_fit_s-x_cen+xp
            sources_df.at[s, 'ycentroid_fit'] = ycen_fit_s-y_cen+yp
            sources_df.at[s, 'fwhm(pixels)'] = fwhm_fit
            sources_df.at[s, 'flux_fit'] = flux_fit
            sources_df.at[s, 'mag_fit'] = mag_fit

            if plot_star_cutouts:
                plt.imshow(im, cmap='gray', origin='lower')
                plt.scatter(x_cen, y_cen, marker='x', color='white')
                fit_x, fit_y = (xcen_fit_s, ycen_fit_s)
                plt.scatter(fit_x, fit_y, marker='x', color='red')
                aperture = CircularAperture((fit_x, fit_y), r=fwhm_fit)
                aperture.plot(color='red', lw=1.5)
                plt.text(1, 2, 'fwhm:'+str(np.round(fwhm_fit, 2)),
                         fontsize=12, color='white')
                plt.text(1, 4, 'peak:'+str(np.round(fit_g.amplitude.value, 2)),
                         fontsize=12, color='white')
                plt.text(1, 6, 'mag:'+str(np.round(mag_fit, 2)),
                         fontsize=12, color='white')
                plt.title(s)
                plt.show()

        # excludes if too close to edge to make cutout
        else:
            sources_df.at[s, 'xcentroid_fit'] = np.NaN
            sources_df.at[s, 'ycentroid_fit'] = np.NaN
            sources_df.at[s, 'fwhm(pixels)'] = np.NaN
            sources_df.at[s, 'flux_fit'] = np.NaN
            sources_df.at[s, 'mag_fit'] = np.NaN

    return sources_df.copy()


def PanSTARRS_info(table='mean', release='dr1',check_cols=None,
                   check_params=None):

    baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs"
    
    #checks if the table and release params are valid
    #raises value error if not
    releaselist = ("dr1", "dr2")
    if release not in ("dr1","dr2"):
        raise ValueError("Bad value for release (must be one of {})".format(', '.join(releaselist)))
    if release=="dr1":
        tablelist = ("mean", "stack")
    else:
        tablelist = ("mean", "stack", "detection")
    if table not in tablelist:
        raise ValueError("Bad value for table (for {} must be one of {})".format(release, ", ".join(tablelist)))
    
    meta_url = "{baseurl}/{release}/{table}/metadata".format(**locals())
    r_meta = requests.get(meta_url)
    r_meta.raise_for_status()
    meta_json = r_meta.json()
    table_cols_df = pd.DataFrame.from_records(meta_json)

    table_cols = table_cols_df['name'].values
    valid_cols = [c.lower() for c in list(table_cols)]
    
    if isinstance(check_params, dict):
        bad_params = []
        for i in check_params:
            p = i.split('.')[0]
            try:
                i_ind = valid_cols.index(p.lower())
            except ValueError:
                bad_params.append(p)
        if len(bad_params) > 0:
           print('WARNING: params '+str(bad_params)+' are not found in table')
           
    elif check_params is not None:
        raise ValueError('check_params must be None or dictionary')
    
    if isinstance(check_cols, list):
        #checks that columns are in the table/release requested
        bad_cols = []
        good_cols = []
        for i in check_cols:
            try:
                i_ind = valid_cols.index(i.lower())
                good_cols.append(i_ind)
            except ValueError:
                bad_cols.append(i)
        if len(bad_cols) > 0:
            raise ValueError('Some columns not found in table: {}'.format(', '.join(bad_cols)))
        else:
            return table_cols_df.iloc[good_cols]
        
    elif check_cols is None:
        return table_cols_df
    
    else:
        raise ValueError('check_cols must be None or list type')


def PanSTARRS_query(ra, dec, radius, table ='mean', release='dr1',
                    columns=None, params={'nDetections.min': 0}):

    baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs"

    cols_df = PanSTARRS_info(table=table, release=release, check_cols=columns, 
                             check_params=params)
    col_list = list(cols_df['name'].values)

    data = {'ra':ra, 'dec':dec, 'radius':radius}
    data = {**data, **params}
    data['columns'] = '[{}]'.format(','.join(col_list))
    print(data)

    url = "{baseurl}/{release}/{table}.csv".format(**locals())
    r = requests.get(url, params=data)
    r.raise_for_status()
    cat_text = r.text
    cat_tab = ascii.read(cat_text)
    cat_df = cat_tab.to_pandas()
    return cat_df
    

            
    
    
