import numpy as np
import emcee
import math
import string as st
import os
import pandas as pd
from scipy.integrate import quad
import matplotlib.pyplot as plt
import corner
import model_line_functions as mlf

class MCMC_functions():
	def __init__(self, num, line_ID, model_bounds, args, fit_cont):
		self.num          = num
		self.line_ID      = line_ID
		self.model_bounds = model_bounds
		self.args         = args
		self.fit_cont     = fit_cont
		self.lines        = mlf.line_dict[self.line_ID]['lines']
		self.wl_lines     = [int(w.split(']')[1]) for w in self.lines]

		if self.fit_cont:
			self.model    = mlf.line_dict[self.line_ID]['cont_mod']
		else:
			self.model    = mlf.line_dict[self.line_ID]['mod']

		self.int_flux     = 0.0
		self.mc_results   = 0.0
		self.flat_samples = 0.0
		self.flux         = 0.0
		self.flux_err_up  = []
		self.flux_err_lo  = []

	#define the log likelihood function
	def lnlike(self, theta, x, y, yerr):
		mod = self.model(x, theta)
		return -0.5*sum(((y - mod)**2)/(yerr**2))

	#define the log prior function
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

	#define log postierior to sovle with emcee
	def lnprob(self, theta, x, y, yerr):
		lp = self.lnprior(theta)
		if not np.isfinite(lp):
			return -np.inf
		return lp + self.lnlike(theta, x, y, yerr)

	def run_emcee(self, ndim, nwalkers, nchains, thetaGuess):

		pos = [thetaGuess + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
		sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, args=self.args, a=1)

		#print "Burning in ..."
		pos, prob, state = sampler.run_mcmc(pos, nchains[0])

		sampler.reset()

		#print "Running MCMC ..."
		pos, prob, state = sampler.run_mcmc(pos, nchains[0], rstate0=state)

		self.flat_samples = sampler.flatchain
		samples = sampler.chain[:, :, :].reshape((-1, ndim))
		mc_results = [(v[1], v[2]-v[1], v[1]-v[0]) for v in zip(*np.percentile(samples, [16, 50, 84], axis=0))]

		self.mc_results = mc_results

		#print 'MC RESULTS: ',mc_results 

		return self.flat_samples, mc_results

	def gaussian(self, x, z, sig, inten):
		mu = 100*(1+z)
		return inten * (np.exp(-0.5*np.power(x - mu, 2.) / (np.power(sig, 2.))))

	def integrate_flux(self, model_vals):
		mu = 100*(1+model_vals[0])

		tot_flux = []
		#tot_flux_err= []

		I= model_vals[2] * model_vals[1] * np.sqrt(2*np.pi)
		#print 'Flux: ', model_vals[2], model_vals[1], I
		tot_flux.append(I)

		if self.line_ID != 'NeIII':
			I2 = model_vals[3] * model_vals[1] * np.sqrt(2*np.pi)
			#print 'Flux: ', model_vals[3], model_vals[1], I2
			tot_flux.append(I2)

		#print tot_flux
		return tot_flux

	def calculate_flux(self):
		sol    = [i[0] for i in self.mc_results]
		up_lim = [i[1] for i in self.mc_results]
		lo_lim = [i[2] for i in self.mc_results]

		flux = self.integrate_flux(sol)
		f_up = self.integrate_flux(np.add(sol, up_lim))
		f_lo = self.integrate_flux(np.subtract(sol, lo_lim))

		self.flux = flux
		self.flux_err_up = [f_up[i] - flux[i] for i in range(len(flux))]
		self.flux_err_lo = [flux[i] - f_lo[i] for i in range(len(flux))]
		# for i in range(len(flux)):
		# 	self.flux_err_up.append(np.sqrt((np.subtract(f_up[i], flux[i])**2) + (int_err[i]**2) + (int_err_up[i]**2)))
		# 	self.flux_err_lo.append(np.sqrt((np.subtract(flux[i], f_lo[i])**2) + (int_err[i]**2) + (int_err_lo[i]**2)))

		return self.flux, (self.flux_err_up, self.flux_err_lo)

	def plot_results(self, name=None, corner_plot=False, flux_factor=-16):
		wl_sol, dat, disp = self.args

		sol    = [i[0] for i in self.mc_results]
		up_lim = [i[1] for i in self.mc_results]
		lo_lim = [i[2] for i in self.mc_results]

		#print sol

		up_lim_theta = np.add(sol,up_lim)
		lo_lim_theta = np.subtract(sol,lo_lim)

		if corner_plot:

			if self.line_ID == 'OII_doub':
				labels_lis = ["$z$", "$simga$", "$[OII] \lambda 3726 intensity$", "[OII] \lambda 3729 $intensity$", "$continuum slope$", "$continuum y-intercept$"]
			if self.line_ID == 'NeIII':
				labels_lis = ["$z$", "$simga$", "$[NeIII] \lambda 3869 intensity$", "$continuum slope$", "$continuum y-intercept$"]
			if self.line_ID == 'OIII_Hb_trip':
				labels_lis = ["$z$", "$simga$", "$H\beta intensity$", "[OIII] \lambda 5007 $intensity$", "$continuum slope$", "$continuum y-intercept$"]

			fig = corner.corner(self.flat_samples, labels=labels_lis, truths=sol, figsize=(1, 1))
			#plt.pause(0.01)
			#plt.close()
			plt.savefig(name+'_cornerplot.png')
			#plt.show()



		else:
			fig, ax = plt.subplots()
			#fig.set_size_inches(20,9)
			#fig.set_size_inches(7,4)

			#print '\n'
			#print 'len flux ', self.flux

			finer_wl_sol = np.linspace(np.amin(wl_sol), np.amax(wl_sol), num=len(wl_sol)*5)
			for rand_theta in self.flat_samples[np.random.randint(len(self.flat_samples), size=300)]:
				ax.plot(finer_wl_sol, self.model(finer_wl_sol, theta=rand_theta), color='grey', alpha=0.25)
			ax.plot(wl_sol, dat)
			ax.plot(finer_wl_sol, self.model(finer_wl_sol, theta = sol), color='red')
			#plt.plot(wl_sol, self.model(wl_sol, theta = up_lim_theta), color='red', ls=':')
			#plt.plot(wl_sol, self.model(wl_sol, theta = lo_lim_theta), color='red', ls=':')

			if self.line_ID == 'OIII_Hb_trip':
				y_vals = np.linspace(np.min(dat), np.amax(dat)+0.5, 50)
				ax.fill_betweenx(y_vals, np.zeros(50)+5566, np.zeros(50)+5590, color='grey', alpha=0.4, zorder=1000)
        

			res_dict = {'flux': '{'+str(round(self.flux[0],2))+'}', 'up_err':'{+'+str(round(self.flux_err_up[0],2))+'}', 'lo_err':'{-'+str(round(self.flux_err_lo[0],2))+'}'}
			sca_dict = {'scale':flux_factor}
			x_pos, y_pos = (0.51, 0.86) #(0.05, 0.86), (0.51, 0.86)
			ax.text(x_pos, y_pos, str(self.lines[0])+r' Flux: (${flux}^{up_err}_{lo_err}$)'.format(**res_dict)+r'$x10^{-17}$',
				transform = ax.transAxes, color='black', alpha=0.8, weight='bold', size=30, bbox=dict(facecolor='whitesmoke', edgecolor='darkgrey', pad=10.0), zorder=2000)

			if (self.line_ID != 'NeIII') and (self.line_ID != 'OII') and (self.line_ID != 'OIII_Te') and (self.line_ID != 'OI'):
				res_dict = {'flux': '{'+str(round(self.flux[1],2))+'}', 'up_err':'{+'+str(round(self.flux_err_up[1],2))+'}', 'lo_err':'{-'+str(round(self.flux_err_lo[1],2))+'}'}
				x_pos2, y_pos2 = (0.51, 0.68) #(0.05, 0.68), (0.51, 0.68) 
				ax.text(x_pos2, y_pos2, str(self.lines[1])+r' Flux: (${flux}^{up_err}_{lo_err}$)'.format(**res_dict)+r'$x10^{-17}$', 
					transform = ax.transAxes, color='black', alpha=0.8, weight='bold', size=30, bbox=dict(facecolor='whitesmoke', edgecolor='darkgrey', pad=10.0), zorder=2001)

			ax.set_ylabel(r'$Flux\ (ergs/s/cm^2/\AA)\ (x{10}^{-17})$', weight='bold', fontsize=15)
			ax.set_xlabel(r'$Wavelength\ (\AA)$', weight='bold', fontsize=15)
			plt.tick_params(axis='both', which='major', labelsize=15)
			#ax.set_ylim(-0.2, 1.0)

			#Set plot labels
			if name == None:
				ax.set_title('Spectrum Fit: '+str(self.num), weight='bold', fontsize=15)
				#sets plotting speed and closes the plot before opening a new one
				plt.show()
				plt.pause(0.01)
				plt.close()

			else:
				#plt.title('Spectrum Fit: '+str(name), fontsize=20)
				print('saving plot')
				plt.savefig(name+'.pdf', bbox_inches='tight')	
				plt.close()

	def write_results(self, df, ind):
		for l in range(len(self.lines)):
			col   = self.lines[l]
			col_e = self.lines[l]+'_e'
			row   = ind
			val   = np.round(self.flux[l],3)
			val_e = str([np.round(self.flux_err_up[l],3), np.round(self.flux_err_lo[l],3)])

			df.at[row, col] = val
			df.at[row, col_e] = val_e
		return df



