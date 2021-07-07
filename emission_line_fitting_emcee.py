#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 15:28:36 2021

@author: Briana
"""

import numpy as np
import emcee
import pandas as pd
import model_line_functions as mlf


class MCMC_functions():

    def __init__(self, num, line_ID, model_bounds, args, fit_cont):
        self.num = num
        self.line_ID = line_ID
        self.model_bounds = model_bounds
        self.args = args
        self.fit_cont = fit_cont
        self.lines = mlf.line_dict[self.line_ID]['lines']
        self.wl_lines = [int(w.split(']')[1]) for w in self.lines]

        if self.fit_cont:
            self.model = mlf.line_dict[self.line_ID]['cont_mod']
        else:
            self.model = mlf.line_dict[self.line_ID]['mod']

        self.int_flux = 0.0
        self.mc_results = 0.0
        self.flat_samples = 0.0
        self.flux = 0.0
        self.flux_err_up = []
        self.flux_err_lo = []

    # define the log likelihood function
    def lnlike(self, theta, x, y, yerr):
        mod = self.model(x, theta)
        return -0.5*sum(((y - mod)**2)/(yerr**2))

    # define the log prior function
    def lnprior(self, theta):
        if self.fit_cont:
            if len(theta) == 5:
                if (self.model_bounds[0][0] <= theta[0] <= self.model_bounds[0][1]) and (self.model_bounds[1][0] <= theta[1] <= self.model_bounds[1][1]) and (self.model_bounds[2][0] <= theta[2] <= self.model_bounds[2][1]) and (self.model_bounds[3][0] <= theta[3] <= self.model_bounds[3][1]) and (self.model_bounds[4][0] <= theta[4] <= self.model_bounds[4][1]):
                    return 0.0
                return -np.inf
            elif len(theta) == 6:
                if (self.model_bounds[0][0] <= theta[0] <= self.model_bounds[0][1]) and (self.model_bounds[1][0] <= theta[1] <= self.model_bounds[1][1]) and (self.model_bounds[2][0] <= theta[2] <= self.model_bounds[2][1]) and (self.model_bounds[3][0] <= theta[3] <= self.model_bounds[3][1]) and (self.model_bounds[4][0] <= theta[4] <= self.model_bounds[4][1]) and (self.model_bounds[5][0] <= theta[5] <= self.model_bounds[5][1]):
                    return 0.0
                return -np.inf
        else:
            if len(theta) == 3:
                if (self.model_bounds[0][0] <= theta[0] <= self.model_bounds[0][1]) and (self.model_bounds[1][0] <= theta[1] <= self.model_bounds[1][1]) and (self.model_bounds[2][0] <= theta[2] <= self.model_bounds[2][1]):
                    return 0.0
                return -np.inf
            elif len(theta) == 4:
                if (self.model_bounds[0][0] <= theta[0] <= self.model_bounds[0][1]) and (self.model_bounds[1][0] <= theta[1] <= self.model_bounds[1][1]) and (self.model_bounds[2][0] <= theta[2] <= self.model_bounds[2][1]) and (self.model_bounds[3][0] <= theta[3] <= self.model_bounds[3][1]):
                    return 0.0
                return -np.inf

    # define log postierior to sovle with emcee
    def lnprob(self, theta, x, y, yerr):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta, x, y, yerr)

    def run_emcee(self, ndim, nwalkers, nchains, thetaGuess):

        pos = [thetaGuess + 1e-4*np.random.randn(ndim) for i in
               range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob,
                                        args=self.args, a=1)

        pos, prob, state = sampler.run_mcmc(pos, nchains[0])
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(pos, nchains[0], rstate0=state)

        self.flat_samples = sampler.flatchain
        samples = sampler.chain[:, :, :].reshape((-1, ndim))
        mc_results = [(v[1], v[2]-v[1], v[1]-v[0]) for v in
                      zip(*np.percentile(samples, [16, 50, 84], axis=0))]

        self.mc_results = mc_results

        return self.flat_samples, mc_results

    def gaussian(self, x, z, sig, inten):
        mu = 100*(1+z)
        return inten * (np.exp(-0.5*np.power(x - mu, 2.) /
                               (np.power(sig, 2.))))

    def integrate_flux(self, model_vals):
        tot_flux = []
        line_flux1 = model_vals[2] * model_vals[1] * np.sqrt(2*np.pi)
        tot_flux.append(line_flux1)

        if self.line_ID != 'NeIII':
            line_flux2 = model_vals[3] * model_vals[1] * np.sqrt(2*np.pi)
            tot_flux.append(line_flux2)

        return tot_flux

    def calculate_flux(self):
        sol = [i[0] for i in self.mc_results]
        up_lim = [i[1] for i in self.mc_results]
        lo_lim = [i[2] for i in self.mc_results]

        flux = self.integrate_flux(sol)
        f_up = self.integrate_flux(np.add(sol, up_lim))
        f_lo = self.integrate_flux(np.subtract(sol, lo_lim))

        self.flux = flux
        self.flux_err_up = [f_up[i] - flux[i] for i in range(len(flux))]
        self.flux_err_lo = [flux[i] - f_lo[i] for i in range(len(flux))]

        return self.flux, (self.flux_err_up, self.flux_err_lo)

    def write_results(self, df, ind):
        for l in range(len(self.lines)):
            col = self.lines[l]
            col_e = self.lines[l]+'_e'
            row = ind
            val = np.round(self.flux[l], 3)
            val_e = str([np.round(self.flux_err_up[l], 3),
                         np.round(self.flux_err_lo[l], 3)])

            df.at[row, col] = val
            df.at[row, col_e] = val_e
        return df
