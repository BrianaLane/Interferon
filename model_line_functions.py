import numpy as np
import string as st

#*****************************#
# Continuum Subtracted Models #
#*****************************#

def OI_gaussian(x, theta):
	z, sig, inten = theta
	mu = 6300*(1+z)
	return inten * (np.exp(-0.5*np.power(x - mu, 2.) / (np.power(sig, 2.))))

def OII_gaussian(x, theta):
	z, sig, inten = theta
	mu = 3727*(1+z)
	return inten * (np.exp(-0.5*np.power(x - mu, 2.) / (np.power(sig, 2.))))

def OIII_Te_gaussian(x, theta):
	z, sig, inten = theta
	mu = 4363*(1+z)
	return inten * (np.exp(-0.5*np.power(x - mu, 2.) / (np.power(sig, 2.))))

def Hb_gaussian(x, theta):
	z, sig, inten = theta
	mu = 4861*(1+z)
	return inten * (np.exp(-0.5*np.power(x - mu, 2.) / (np.power(sig, 2.))))

def NeIII_gaussian(x, theta):
	z, sig, inten = theta
	mu = 3870*(1+z)
	return inten * (np.exp(-0.5*np.power(x - mu, 2.) / (np.power(sig, 2.))))

#fit the OIII doublet and fixes the ratio to 2.89
def OII_doub_gaussian(x, theta):
	z, sig, inten1, inten2 = theta
	mu1 = 3726*(1+z)
	mu2 = 3729*(1+z)
	return (inten1 * (np.exp(-0.5*np.power(x - mu1, 2.) / (np.power(sig, 2.))))) + \
		(inten2 * (np.exp(-0.5*np.power(x - mu2, 2.) / (np.power(sig, 2.)))))

#fit the OIII doublet and fixes the ratio to 2.89
def OIII_doub_gaussian(x, theta):
	z, sig, inten = theta
	mu1 = 5007*(1+z)
	mu2 = 4959*(1+z)
	return (inten * (np.exp(-0.5*np.power(x - mu1, 2.) / (np.power(sig, 2.))))) + \
		((inten/2.98) * (np.exp(-0.5*np.power(x - mu2, 2.) / (np.power(sig, 2.)))))

#fits independent SII doublet, no fixed ratio so get 2 intensity values
def SII_doub_gaussian(x, theta):
	z, sig, inten1, inten2 = theta
	mu1 = 6731*(1+z)
	mu2 = 6717*(1+z)
	return (inten1 * (np.exp(-0.5*np.power(x - mu1, 2.) / (np.power(sig, 2.))))) + \
		(inten2 * (np.exp(-0.5*np.power(x - mu2, 2.) / (np.power(sig, 2.)))))

#fit the OIII doublet and fixes the ratio to 2.89. Also fits Hb which is blueward of the doublet 
def OIII_Hb_trip_gaussian(x, theta):
	z, sig, inten1, inten2 = theta
	mu1 = 5007*(1+z)
	mu2 = 4959*(1+z)
	mu3 = 4861*(1+z)
	return (inten1 * (np.exp(-0.5*np.power(x - mu1, 2.) / (np.power(sig, 2.))))) + \
		((inten1/2.98) * (np.exp(-0.5*np.power(x - mu2, 2.) / (np.power(sig, 2.))))) + \
		(inten2 * (np.exp(-0.5*np.power(x - mu3, 2.) / (np.power(sig, 2.)))))

#fit the NII doublet and fixes the ratio to 2.5 and also fits Ha which is between the lines 
def NII_Ha_trip_gaussian(x, theta):
	z, sig, inten1, inten2 = theta
	mu1 = 6583*(1+z)
	mu2 = 6562*(1+z)
	mu3 = 6549*(1+z)
	return (inten1 * (np.exp(-0.5*np.power(x - mu1, 2.) / (np.power(sig, 2.))))) + \
		(inten2 * (np.exp(-0.5*np.power(x - mu2, 2.) / (np.power(sig, 2.))))) + \
		((inten1/3.0) * (np.exp(-0.5*np.power(x - mu3, 2.) / (np.power(sig, 2.)))))

#************************#
# Models with Continuum  #
#************************#

def OI_gaussian_cont(x, theta):
	z, sig, inten, m, b = theta
	mu = 6300*(1+z)
	return ((m*x)+b) + inten * (np.exp(-0.5*np.power(x - mu, 2.) / (np.power(sig, 2.))))

def OII_gaussian_cont(x, theta):
	z, sig, inten, m, b = theta
	mu = 3727*(1+z)
	return ((m*x)+b) + inten * (np.exp(-0.5*np.power(x - mu, 2.) / (np.power(sig, 2.))))

def OIII_Te_gaussian_cont(x, theta):
	z, sig, inten, m, b = theta
	mu = 4363*(1+z)
	return ((m*x)+b) + inten * (np.exp(-0.5*np.power(x - mu, 2.) / (np.power(sig, 2.))))

def Hb_gaussian_cont(x, theta):
	z, sig, inten, m, b = theta
	mu = 4861*(1+z)
	return ((m*x)+b) + inten * (np.exp(-0.5*np.power(x - mu, 2.) / (np.power(sig, 2.))))

def NeIII_gaussian_cont(x, theta):
	z, sig, inten, m, b = theta
	mu = 3870*(1+z)
	return ((m*x)+b) + inten * (np.exp(-0.5*np.power(x - mu, 2.) / (np.power(sig, 2.))))

#fit the OIII doublet and fixes the ratio to 2.89
def OII_doub_gaussian_cont(x, theta):
	z, sig, inten1, inten2, m, b = theta
	mu1 = 3726*(1+z)
	mu2 = 3729*(1+z)
	return ((m*x)+b) + (inten1 * (np.exp(-0.5*np.power(x - mu1, 2.) / (np.power(sig, 2.))))) + \
		(inten2 * (np.exp(-0.5*np.power(x - mu2, 2.) / (np.power(sig, 2.)))))

#fit the OIII doublet and fixes the ratio to 2.89
def OIII_doub_gaussian_cont(x, theta):
	z, sig, inten, m, b = theta
	mu1 = 5007*(1+z)
	mu2 = 4959*(1+z)
	return ((m*x)+b) + (inten * (np.exp(-0.5*np.power(x - mu1, 2.) / (np.power(sig, 2.))))) + \
		((inten/2.98) * (np.exp(-0.5*np.power(x - mu2, 2.) / (np.power(sig, 2.)))))

#fits independent SII doublet, no fixed ratio so get 2 intensity values
def SII_doub_gaussian_cont(x, theta):
	z, sig, inten1, inten2, m, b = theta
	mu1 = 6731*(1+z)
	mu2 = 6717*(1+z)
	return ((m*x)+b) + (inten1 * (np.exp(-0.5*np.power(x - mu1, 2.) / (np.power(sig, 2.))))) + \
		(inten2 * (np.exp(-0.5*np.power(x - mu2, 2.) / (np.power(sig, 2.)))))

#fit the OIII doublet and fixes the ratio to 2.89. Also fits Hb which is blueward of the doublet 
def OIII_Hb_trip_gaussian_cont(x, theta):
	z, sig, inten1, inten2, m, b = theta
	mu1 = 5007*(1+z)
	mu2 = 4959*(1+z)
	mu3 = 4861*(1+z)
	return ((m*x)+b) + (inten1 * (np.exp(-0.5*np.power(x - mu1, 2.) / (np.power(sig, 2.))))) + \
		((inten1/2.98) * (np.exp(-0.5*np.power(x - mu2, 2.) / (np.power(sig, 2.))))) + \
		(inten2 * (np.exp(-0.5*np.power(x - mu3, 2.) / (np.power(sig, 2.)))))

#fit the NII doublet and fixes the ratio to 2.5 and also fits Ha which is between the lines 
def NII_Ha_trip_gaussian_cont(x, theta):
	z, sig, inten1, inten2, m, b = theta
	mu1 = 6583*(1+z)
	mu2 = 6562*(1+z)
	mu3 = 6549*(1+z)
	return ((m*x)+b) + (inten1 * (np.exp(-0.5*np.power(x - mu1, 2.) / (np.power(sig, 2.))))) + \
		(inten2 * (np.exp(-0.5*np.power(x - mu2, 2.) / (np.power(sig, 2.))))) + \
		((inten1/2.5) * (np.exp(-0.5*np.power(x - mu3, 2.) / (np.power(sig, 2.)))))

#********************#
# Trim Spec Function #
#********************#

line_dict = {'OI':				{'mod':OI_gaussian,				'cont_mod':OI_gaussian_cont,				'lines':['[OI]6300'],				'trim':(6270.0, 6330.0)},
			'OII':				{'mod':OII_gaussian,			'cont_mod':OII_gaussian_cont,				'lines':['[OII]3727'],				'trim':(3700.0, 3780.0)},
			'OIII_Te':			{'mod':OIII_Te_gaussian,		'cont_mod':OIII_Te_gaussian_cont,			'lines':['[OIII]4363'],				'trim':(4330.0, 4390.0)},
			'Hb':				{'mod':Hb_gaussian,				'cont_mod':Hb_gaussian_cont,				'lines':['[Hb]4861'],				'trim':(4830.0, 4890.0)},
			'NeIII':			{'mod':NeIII_gaussian,			'cont_mod':NeIII_gaussian_cont,				'lines':['[NeIII]3870'],			'trim':(3840.0, 3900.0)},
			'OII_doub':			{'mod':OII_doub_gaussian,		'cont_mod':OII_doub_gaussian_cont,			'lines':['[OII]3726','[OII]3729'],	'trim':(3700.0, 3760.0)},
			'OIII_doub':		{'mod':OIII_doub_gaussian,		'cont_mod':OIII_doub_gaussian_cont,			'lines':['[OIII]5007'],				'trim':(4930.0, 5050.0)},
			'SII_doub':			{'mod':SII_doub_gaussian,		'cont_mod':SII_doub_gaussian_cont,			'lines':['[SII]6731','[SII]6717'],	'trim':(6680.0, 6780.0)},
			'OIII_Hb_trip':		{'mod':OIII_Hb_trip_gaussian,	'cont_mod':OIII_Hb_trip_gaussian_cont,		'lines':['[OIII]5007','[Hb]4861'],	'trim':(4800.0, 5100.0)},
			'NII_Ha_trip':		{'mod':NII_Ha_trip_gaussian,	'cont_mod':NII_Ha_trip_gaussian_cont,		'lines':['[NII]6583','[Ha]6562'],	'trim':(6500.0, 6630.0)}}

def trim_spec_for_model(line, dat, residuals, wl, z):
	min_wave = line_dict[line]['trim'][0]*(1+z)
	max_wave = line_dict[line]['trim'][1]*(1+z)
	inds = np.where((min_wave < wl) & (wl < max_wave))

	wl = wl[inds]

	if len(np.shape(dat)) > 1:
		dat       = dat[:, inds]
		residuals = residuals[:, inds]
		return np.vstack(dat), np.vstack(residuals), wl
	else:
		dat       = dat[inds]
		residuals = residuals[inds]
		return dat, residuals, wl
