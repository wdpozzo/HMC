from astropy import constants as const
M_sun = const.M_sun.value
G = const.G.value
c = const.c.value
pc = const.pc.value
import jax.numpy as jnp
import numpy as np
from functools import partial
import jax
from jax import jacrev, grad, jacobian, vjp
from jax.tree_util import tree_map
from utils import GreenwichMeanSiderealTime

import sys
import ray
from utils import TimeDelayFromEarthCenter

from granite.powerspectrum.mesa import psd_onsource
from granite.noise.noise import load_data

from hmc import HamiltonianMonteCarlo, NUTS

def TaylorF2(params, frequency_array):
    
    Mc = params["mc"]
    q  = params["q"]
    phi_c = params["phiref"],
    distance = jnp.exp(params["logdistance"])
    iota = jnp.arccos(params["costheta_jn"])
    
    nu = q/((1+q)**2)
    f_max = frequency_array[-1]

    r = distance
    #print(pc)
    Mc = Mc*M_sun
    r = r*pc*1e6#megaparsec
    M = Mc/(nu**(3/5))

    f_lso =f_max/2
    #nu = m1*m2/(m1+m2)**2
    v = (G*jnp.pi*M*frequency_array)**(1/3)
    #print(v)
    v = v/c
    
    v_lso = (G*jnp.pi*M*f_lso)**(1/3)
    v_lso = v_lso /c
    
    
    gamma  = jnp.euler_gamma

    
    amp =(jnp.pi**(-2/3))*jnp.sqrt(5/24)*(G*Mc/(c**3))**(5/6)*frequency_array**(-7/6)*(c/r)###questo è ok
    phi_plus=  ((3)/(128*nu*(v)**5))*(1
    
    +
    #0PN
    v**2*(20/9)*(743/336 + nu*11/4)
    -\
    #1PN
    16*jnp.pi*v**3+ \
    #1.5PN
    10*(v**4)*(3058673/1016064 +nu*5429/1008+(nu**2)*617/144)
    +\
    #2PN
    v**(5)*jnp.pi*(38645/756-nu*65/9)*(1+3*jnp.log(v))
    +\
    #2.5PN
    (v**6)*(11583231236531/4694215680 -jnp.pi**2*640/3 -6848*gamma/21 - 6848/21*jnp.log(4*v)+\

    nu*(-15737765635/3048192+ 2255*(jnp.pi**2)/12)+nu**2*76055/1728 -nu**3*127825/1296)+\
    #3PN
    v**(7)*jnp.pi*(77096675/254016 +nu*378515/1512 -(nu**2)*74045/756))


    phi_plus += jnp.pi- jnp.pi/4


    phi_cross = phi_plus  + jnp.pi/2
    #phi= jnp.exp(1j*(-(phi_c) + 2*jnp.pi*frequency_array*t_c ))
    phi= jnp.exp(1j*(-(params["phiref"])))
    h_plus = phi*amp*((1+(jnp.cos(iota))**2)/2)*jnp.exp(1j*phi_plus)
    h_cross = phi*amp*jnp.cos(iota)*jnp.exp(1j*phi_cross)
    '''
    f_tot =jnp.arange(0, f_max, delta_f)
    h_pad = jnp.zeros(len(f_tot)-len(f))
    h_plus = jnp.concatenate((h_pad, h_plus))
    h_cross = jnp.concatenate((h_pad, h_cross))
    '''
    return h_plus, h_cross

class GWDetector:
    """
    Class for a gravitational wave detector

    Arguments
    ---------
    name : string
        Name of the detector. Use GWDetector.get_detector_name to see a list of available values.
    """

    # in order, we have:
    # [latitude, longitude, orientation of detector arms, angle between the arms]
    available_detectors = {
        'V1': [43.63, 10.5, 115.56, 90.],
        'H1': [46.45, -119.41, 170.9, 90.],
        'L1': [30.56, -90.77, 242.7, 90.],
        'GEO600': [52.25, -9.81, 68.775, 94.33],
        'TAMA300': [35.68, -139.54, 225., 90.],
        'ET': [40.44, 9.4566, 116.5, 60.], # Sardinia site hypothesis
        'K': [36.41, 137.30, 15.36, 90.]
    }


    def __init__(self,
                 name,
                 datafile           = None,
                 psd_file           = None,
                 psd_method         = 'mesa-on-source',
                 T                  = 2.0,
                 starttime          = 1126259461.423,
                 trigtime           = 1126259462.423,
                 sampling_rate      = 1024.,
                 flow               = 20,
                 fhigh              = None,
                 zero_noise         = False,
                 calibration        = None,
                 download_data      = 1,
                 datalen_download   = 32,
                 channel            = '',
                 gwpy_tag           = None):
                 
        # initialise the needed attributes
        self.name             = name
        self.latitude         = self.available_detectors[name][0]
        self.longitude        = self.available_detectors[name][1]
    
        if name not in self.available_detectors.keys():
            raise ValueError("Not valid argument ({}) for 'name' parameter.".format(name))

        self.datafile         = datafile
        self.psd_file         = psd_file
        self.psd_method       = psd_method
        self.Epoch            = jnp.float64(starttime)
        self.sampling_rate    = sampling_rate
        self.flow             = flow
        self.trigtime         = trigtime
        self.zero_noise       = zero_noise
        self.calibration      = calibration
        self.T                = T
        self.download_data    = download_data
        self.datalen_download = datalen_download
        self.channel          = channel
        self.gwpy_tag         = gwpy_tag

        self.Times, self.TimeSeries, self.Frequency, self.FrequencySeries, self.PowerSpectralDensity, self.mesa_object = load_data(self.datafile,
                                         self.name,
                                         chunk_size       = self.T,
                                         trigtime         = self.trigtime,
                                         sampling_rate    = self.sampling_rate,
                                         psd_file         = self.psd_file,
                                         psd_method       = self.psd_method,
                                         download_data    = self.download_data,
                                         datalen_download = self.datalen_download,
                                         channel          = self.channel,
                                         gwpy_tag         = self.gwpy_tag)

        # set the maximum frequency cutoff to prevent aliasing
        if fhigh is None:
            self.fhigh = self.sampling_rate*0.45
            sys.stdout.write('\nMaximum frequency not given: it will be set to sampling_rate*0.45 to prevent aliasing\n')
        elif fhigh>self.sampling_rate/2.:
            self.fhigh = self.sampling_rate*0.45
            sys.stdout.write('\nMaximum frequency above the Nyquist bound: it will be set to sampling_rate*0.45 to prevent aliasing\n')
        else:
            self.fhigh = fhigh
            
        # set frequency-related specifics
        self.df             = 1./self.T
        self.dt             = 1./self.sampling_rate
        self.segment_length = int(self.T*self.sampling_rate)
        self.kmin           = int(self.flow/self.df)
        self.kmax           = int(self.fhigh/self.df)+1
        
        # crop the frequency series and the frequency array
        self.FrequencySeries      = self.FrequencySeries[self.kmin:self.kmax]
        self.Frequency            = self.Frequency[self.kmin:self.kmax]
        self.PowerSpectralDensity = self.PowerSpectralDensity[self.kmin:self.kmax]
        
        # noise-weighted inner product weighting factor
        self.sigmasq              = self.PowerSpectralDensity * self.dt * self.dt
        self.TwoDeltaTOverN       = 2.0*self.dt/jnp.float64(self.segment_length)

        self.latitude = self.available_detectors[name][0]
        self.longitude = self.available_detectors[name][1]
        self.gamma = self.available_detectors[name][2]
        self.zeta = self.available_detectors[name][3]
        
    @staticmethod
    def _ab_factors(g_, lat, ra, dec, lst):
        """
        Method that calculates the amplitude factors of plus and cross
        polarization in the wave projection on the detector.
        :param g_: float
            this represent the orientation of the detector's arms with respect to local geographical direction, in
            rad. It is measured counterclock-wise from East to the bisector of the interferometer arms.
        :param lat: float
            longitude of the detector in rad.
        :param ra: float
            Right ascension of the source in rad.
        :param dec: float
            Declination of the source in rad.
        :param lst: float or ndarray
            Local sidereal time(s) in rad.
        :return: tuple of float or np.ndarray
            relative amplitudes of hplus and hcross.
        """

        a_ = (1/16)*jnp.sin(2*g_)*(3-jnp.cos(2*lat))*(3-jnp.cos(2*dec))*jnp.cos(2*(ra - lst))-\
             (1/4)*jnp.cos(2*g_)*jnp.sin(lat)*(3-jnp.cos(2*dec))*jnp.sin(2*(ra - lst))+\
             (1/4)*jnp.sin(2*g_)*jnp.sin(2*lat)*jnp.sin(2*dec)*jnp.cos(ra - lst)-\
             (1/2)*jnp.cos(2*g_)*jnp.cos(lat)*jnp.sin(2*dec)*jnp.sin(ra - lst)+\
             (3/4)*jnp.sin(2*g_)*(jnp.cos(lat)**2)*(jnp.cos(dec)**2)

        b_ = jnp.cos(2*g_)*jnp.sin(lat)*jnp.sin(dec)*jnp.cos(2*(ra - lst))+\
             (1/4)*jnp.sin(2*g_)*(3-jnp.cos(2*lat))*jnp.sin(dec)*jnp.sin(2*(ra - lst))+\
             jnp.cos(2*g_)*jnp.cos(lat)*jnp.cos(dec)*jnp.cos(ra - lst)+\
             (1/2)*jnp.sin(2*g_)*jnp.sin(2*lat)*jnp.cos(dec)*jnp.sin(ra - lst)


        return a_, b_
    '''
    def TimeDelayFromEarthCenter():
    
        ehat_src = np.zeros(3)
        greenwich_hour_angle = GreenwichMeanSiderealTime(params["tc"]) - ra

        ehat_src[0] = np.cos(dec) * np.cos(greenwich_hour_angle)
        ehat_src[1] = np.cos(dec) * -np.sin(greenwich_hour_angle)
        ehat_src[2] = np.sin(dec)
        time_delay = 0.0
        for i in range(3):
            time_delay += - ehat_src[i] * detector_location[i]
        return time_delay / c

    '''


    def project_waveform(self, params):
    
        h_plus, h_cross = TaylorF2(params, self.Frequency)
        #gmst = np.radians(self.lst_estimate(GPS_time))
        fplus, fcross   = self.antenna_pattern_functions(params)
        

        timedelay       = TimeDelayFromEarthCenter(self.latitude, self.longitude, params['ra'], params['dec'], params['tc'])
        timeshift       = timedelay
        shift           = 2.0*np.pi*self.Frequency*timeshift

        h = (fplus*h_plus + fcross*h_cross)*(jnp.cos(shift)-1j*jnp.sin(shift))
        return h

    def antenna_pattern_functions(self, params):
        '''
        Evaluate the antenna pattern functions.

        :param right_ascension: float
            Right ascension of the source in degree.

        :param declination: float
            Declination of the source in degree.

        :param polarization: float
            Polarization angle of the wave in degree.

        :param GPS_time: float, int, list or np.ndarray
            time of arrival of the source signal.

        :return: tuple of float or np.ndarray
            fplus and fcross.
        '''

        ra = params["ra"]#np.radians(right_ascension)
        dec = params["dec"]#np.radians(declination)

        pol = params["psi"]#np.radians(polarization)
        lat = jnp.radians(self.latitude)
        g_ = jnp.radians(self.gamma)
        z_ = jnp.radians(self.zeta)
        gmst = jnp.mod(GreenwichMeanSiderealTime(params["tc"]), 2*jnp.pi)
        lst = gmst + jnp.radians(self.longitude)
        ampl11, ampl12 = self._ab_factors(g_, lat, ra, dec, lst)

        
        fplus = jnp.sin(z_)*(ampl11*jnp.cos(2*pol) + ampl12*jnp.sin(2*pol))
        fcross = jnp.sin(z_)*(ampl12*jnp.cos(2*pol) - ampl11*jnp.sin(2*pol))

        return fplus, fcross
    #@partial(jax.jit, static_argnums=(0,))
    def log_likelihood(self, params):
    
        h = self.project_waveform(params)
        
        residuals = self.FrequencySeries - h
        
        return -self.TwoDeltaTOverN*jnp.vdot(residuals, residuals/self.sigmasq).real
    
#    def potential(self, params):
#    
#        return -self.log_posterior(params)
#    
#    def log_prior(self, params):
#    
#        logP = 0.0
#        logP += 3.0*params['logdistance']
#
#        # declination
#        logP += np.log(np.abs(np.cos(params['dec'])))
#
#        # chirp mass and mass ratio
#        mc      = params['mc']
#        q       = params['q']
#        logP += np.log(mc)
#        logP += (2./5.)*np.log(1.0+q)-(6./5.)*np.log(q)
#        return logP
#    
#    def log_posterior(self, params):
#        return self.log_prior(params) + self.log_likelihood(params)
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import jax
    jax.config.update("jax_enable_x64", True)

    from raynest.model import Model
    from scipy.stats import norm
    from raynest.nest2pos import autocorrelation, acl
    
    class RapidPE(Model):
        
        def __init__(self, n, b, detector_names):
            self.names  = n
            self.bounds = []
            for name in self.names:
                self.bounds.append([b[name][0],b[name][1]])
            self.detectors = [GWDetector(det, channel = "GWOSC") for det in detector_names]
        
        def log_prior(self, params):
        
            logP = 0.0
            logP += 3.0*params['logdistance']

            # declination
            logP += jnp.log(jnp.abs(jnp.cos(params['dec'])))

            # chirp mass and mass ratio
            mc      = params['mc']
            q       = params['q']
            logP   += jnp.log(mc)
            logP   += (2./5.)*jnp.log(1.0+q)-(6./5.)*jnp.log(q)
            return logP
        
        def log_posterior(self, params):
            return self.log_prior(params) + self.log_likelihood(params)
        
        def log_likelihood(self, params):
            # Ensure the list of log-likelihoods is a JAX array
            log_likelihoods = jnp.array([det.log_likelihood(params) for det in self.detectors])

            # Then use jnp.sum
            return jnp.sum(log_likelihoods)
        
        def log_posterior(self, q):
            logP = self.log_prior(q)+self.log_likelihood(q)
            return logP

        def potential(self, q):
            return -self.log_posterior(q)
        
        def gradient(self, params):
            """
            we need to compute for each detector
            
            sum_f (2/PSD) Re (h grad h^*) 
            """
            params = tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), dict(params))
            g      = grad(lambda p: self.log_posterior(p))(params)
            #print("parameter =",params)
            #print("gradient =",g)
            #print("posterior =",self.log_posterior(params))
            return g
     
#    ray.init()

    # default parameters' names
    default_names = ['phiref',
                          'ra',
                          'dec',
                          'tc',
                          'mc',
                          'q',
                          'costheta_jn',
                          'psi',
                          'logdistance']

    trigtime = 1126259462.423
    # default prior bounds matching the parameters in self.default_name
    default_bounds = {'phiref'      : [0.0,2.0*jnp.pi],
                           'ra'          : [0.0,2.0*jnp.pi],
                           'dec'         : [-jnp.pi/2.0,jnp.pi/2.0],
                           'tc'          : [trigtime-0.05,trigtime+0.05],
                           'mc'          : [5.0,40.0],
                           'q'           : [0.125,1.0],
                           'costheta_jn' : [-1.0,1.0],
                           'psi'         : [0.0,jnp.pi],
                           'logdistance' : [jnp.log(1.0),jnp.log(2000.0)]}
    
    n_threads  = 1
    n_samps    = 1e5
    n_train    = 1e4
    e_train    = 1
    adapt_mass = 0
    verbose    = 1
    n_bins     = int(np.sqrt(n_samps))
    
    rng       = [np.random.default_rng(1111+j) for j in range(n_threads)]

    M         = RapidPE(default_names, default_bounds, ["H1","L1"])
    
    Kernel    = NUTS
    HMC       = [HamiltonianMonteCarlo(M, Kernel, rng = rng[j], verbose = verbose) for j in range(n_threads)]
    
    samples   = [H.sample(M.new_point(rng = rng[j]),
                          n=n_samps//n_threads,
                          t_epochs=e_train,
                          t=n_train,
                          mass_estimate = adapt_mass,
                          position=j)
                 for j,H in enumerate(HMC)]
    
    x = []
    for s in samples:
        for v in s:
            x.append(v)
    
    v = np.column_stack([[xi[n] for xi in x] for n in names])
    
    import matplotlib.pyplot as plt
    from corner import corner
    corner(v,
                     labels=names,
                     quantiles=[0.05, 0.5, 0.95], truths = None,
                     show_titles=True, title_kwargs={"fontsize": 12}, smooth2d=1.0)
    
    plt.savefig("corner.pdf",bbox_inches='tight')
    

