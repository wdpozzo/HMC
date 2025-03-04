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

from hmc import NUTS

from jax import jit

@jit
def TaylorF2(params, frequency_array):
    # Extract parameters
    Mc, q, phi_c, logdistance, costheta_jn = params[4], params[5], params[0], params[8], params[6]

    # Compute mass and distance-related terms
    distance = jnp.exp(logdistance)
    iota = jnp.arccos(costheta_jn)
    nu = q / ((1 + q) ** 2)

    Mc *= M_sun
    r = distance * pc * 1e6  # Convert to Megaparsec

    M = Mc / (nu ** (3 / 5))
    f_lso = frequency_array[-1] / 2

    # Precompute terms
    pi_M = G * jnp.pi * M
    v = jnp.power(pi_M * frequency_array, 1/3) / c
    v_lso = jnp.power(pi_M * f_lso, 1/3) / c
    gamma = jnp.euler_gamma

    # Compute amplitude
    amp = jnp.power(jnp.pi, -2/3) * jnp.sqrt(5/24) * jnp.power(G * Mc / c**3, 5/6) \
          * jnp.power(frequency_array, -7/6) * (c / r)

    # Compute phase terms (factorized and precomputed where possible)
    v2 = v**2
    v3 = v**3
    v4 = v**4
    v5 = v**5
    v6 = v**6
    v7 = v**7
    log_v = jnp.log(v)

    phi_plus = (3 / (128 * nu * v**5)) * (1 +
        v2 * (20/9) * (743/336 + nu * 11/4) -
        v3 * (16 * jnp.pi) +
        v4 * (10 * (3058673/1016064 + nu * 5429/1008 + (nu**2) * 617/144)) +
        v5 * jnp.pi * (38645/756 - nu * 65/9) * (1 + 3 * log_v) +
        v6 * (11583231236531/4694215680 - jnp.pi**2 * 640/3 - 6848 * gamma/21 - 6848/21 * log_v +
              nu * (-15737765635/3048192 + 2255 * (jnp.pi**2) / 12) + nu**2 * 76055/1728 - nu**3 * 127825/1296) +
        v7 * jnp.pi * (77096675/254016 + nu * 378515/1512 - nu**2 * 74045/756)
    )

    phi_plus += jnp.pi - jnp.pi / 4
    phi_cross = phi_plus + jnp.pi / 2

    # Compute phase factor
    phase_factor = jnp.exp(-1j * phi_c)
    exp_phi_plus = jnp.exp(1j * phi_plus)
    exp_phi_cross = jnp.exp(1j * phi_cross)

    # Compute strain polarizations
    cos_iota = jnp.cos(iota)
    cos_iota_sq = cos_iota**2

    h_plus = phase_factor * amp * ((1 + cos_iota_sq) / 2) * exp_phi_plus
    h_cross = phase_factor * amp * cos_iota * exp_phi_cross

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
    @jit
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

    def project_waveform(self, params):
            #    default_names = ['phiref','ra','dec','tc','mc','q','costheta_jn','psi','logdistance']
        h_plus, h_cross = TaylorF2(params, self.Frequency)
        #gmst = np.radians(self.lst_estimate(GPS_time))
        fplus, fcross   = self.antenna_pattern_functions(params)
        

        timedelay       = TimeDelayFromEarthCenter(self.latitude, self.longitude, params[1], params[2], params[3])
        timeshift       = timedelay
        shift           = 2.0*np.pi*self.Frequency*timeshift

        h = (fplus*h_plus + fcross*h_cross)*(jnp.cos(shift)-1j*jnp.sin(shift))
        return h

    def antenna_pattern_functions(self, params):
        '''
        #    default_names = ['phiref','ra','dec','tc','mc','q','costheta_jn','psi','logdistance']
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

        ra = params[1]#np.radians(right_ascension)
        dec = params[2]#np.radians(declination)

        pol = params[7]#np.radians(polarization)
        lat = jnp.radians(self.latitude)
        g_ = jnp.radians(self.gamma)
        z_ = jnp.radians(self.zeta)
        gmst = jnp.mod(GreenwichMeanSiderealTime(params[3]), 2*jnp.pi)
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
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import jax
    jax.config.update("jax_enable_x64", True)

    from raynest.model import Model
    from scipy.stats import norm
    from raynest.nest2pos import autocorrelation, acl
    
    class RapidPE:
        
        def __init__(self, n, b, detector_names):
            self.names  = n
            self.bounds = b
            self.detectors = [GWDetector(det, channel = "GWOSC") for det in detector_names]
            self.gradient_function = jax.grad(self.log_posterior)
            
        def new_point(self, rng = None):
            """
            Create a new point, drawn from within bounds

            -----------
            Return:
                p: :obj:`np.ndarray`
            """
            
            if rng is None:
                generator = np.random.uniform
            else:
                generator = rng.uniform
            logP = -np.inf
            
            while(logP==-np.inf):
                p = np.array([generator(self.bounds[n][0], self.bounds[n][1]) for n in self.names])
                logP=self.log_prior(p)
            
            return p
        
        def log_prior(self, params):
        #    default_names = ['phiref','ra','dec','tc','mc','q','costheta_jn','psi','logdistance']
        
            logP = 0.0
            logP += 3.0*params[8]

            # declination
            logP += jnp.log(jnp.abs(jnp.cos(params[2])))

            # chirp mass and mass ratio
            mc      = params[4]
            q       = params[5]
            logP   += jnp.log(mc)
            logP   += (2./5.)*jnp.log(1.0+q)-(6./5.)*jnp.log(q)
            return logP
    
        def in_bounds(self,param):
            """
            Checks whether param lies within the bounds

            -----------
            Parameters:
                param: :obj:`raynest.parameter.LivePoint`

            -----------
            Return:
                True: if all dimensions are within the bounds
                False: otherwise
            """
#            for n in param.keys():
#                print(n,"--",self.bounds[n][0],param[n],self.bounds[n][1])
#                if not(self.bounds[n][0] < param[n] < self.bounds[n][1]):
#                    return False
            return all(self.bounds[n][0] < param[n] < self.bounds[n][1] for n in param.keys())
        
        def log_posterior(self, params):
            
            return self.log_likelihood(params) + self.log_prior(params)
        
        def log_likelihood(self, params):
            # Ensure the list of log-likelihoods is a JAX array
            log_likelihoods = jnp.array([det.log_likelihood(params) for det in self.detectors])

            # Then use jnp.sum
            return jnp.sum(log_likelihoods)

        def potential(self, q):
            return -self.log_posterior(q)
        
        def gradient(self, params):
            """
            we need to compute for each detector
            
            sum_f (2/PSD) Re (h grad h^*) 
            """
#            params = tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), dict(params))
#            g      = grad(lambda p: self.log_posterior(p))(params)
#            #g = jax.grad(self.log_posterior)(params.values)
#            #print("parameter =",params)
#            #print("gradient =",g)
#            #print("posterior =",self.log_posterior(params))
#            return g
            return self.gradient_function(params)
     
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
    n_samps    = 1e4
    n_train    = 1e3
    e_train    = 0
    adapt_mass = 0
    verbose    = 1
    n_bins     = int(np.sqrt(n_samps))
    
    rng         = [np.random.default_rng(111+j) for j in range(n_threads)]

    M           = RapidPE(default_names, default_bounds, ["H1","L1"])
    mass_matrix = np.eye(len(default_names))
    
    stds = {'phiref'      : 3.13522148e+00,
           'ra'          : 2.23478705e-01,
           'dec'         : 2.69487404e-02,
           'tc'          : 9.11720777e-06,
           'mc'          : 7.53807316e-01,
           'q'           : 8.31115736e-03,
           'costheta_jn' : 4.63256737e-01,
           'psi'         : 7.80863724e-01,
           'logdistance' : 5.28601392e-02}
#    3.13522148e+00, 2.23478705e-01, 2.69487404e-02, 9.11720777e-06,
#       7.53807316e-01, 8.31115736e-03, 4.63256737e-01, 7.80863724e-01,
#       5.28601392e-02
    for i,v in enumerate(stds.values()):
        mass_matrix[i,i] = 1./v
##
#    print("mass matrix = ",mass_matrix)
#    exit()
    
    mass_matrix = np.linalg.inv(np.array([[ 3.13522148e+00, -6.23718264e-02, -1.10763474e-02,
         3.56164964e-04, -1.14409649e-01, -1.31966170e-04,
         1.69214510e-02,  5.71525977e-02, -1.17262267e-02],
       [-6.23718264e-02,  2.23478705e-01,  2.67315995e-02,
        -1.28340626e-03, -1.20094030e-02,  4.04489879e-03,
        -1.33341820e-02,  3.00868686e-03,  5.51567679e-02],
       [-1.10763474e-02,  2.67315995e-02,  2.69487404e-02,
        -8.68979178e-07, -3.08133331e-03,  5.21338908e-04,
         3.87100007e-02,  5.59990262e-03, -1.06701029e-03],
       [ 3.56164964e-04, -1.28340626e-03, -8.68979178e-07,
         9.11720777e-06, -1.80715855e-06, -1.59490216e-05,
         3.32798566e-04, -2.36703410e-05, -3.59460134e-04],
       [-1.14409649e-01, -1.20094030e-02, -3.08133331e-03,
        -1.80715855e-06,  7.53807316e-01,  1.85505149e-02,
         1.49217049e-02,  1.52733656e-02,  2.77549823e-02],
       [-1.31966170e-04,  4.04489879e-03,  5.21338908e-04,
        -1.59490216e-05,  1.85505149e-02,  8.31115736e-03,
        -3.78035464e-04,  2.96584220e-03,  3.02205777e-03],
       [ 1.69214510e-02, -1.33341820e-02,  3.87100007e-02,
         3.32798566e-04,  1.49217049e-02, -3.78035464e-04,
         4.63256737e-01, -5.84563953e-03, -3.23932733e-02],
       [ 5.71525977e-02,  3.00868686e-03,  5.59990262e-03,
        -2.36703410e-05,  1.52733656e-02,  2.96584220e-03,
        -5.84563953e-03,  7.80863724e-01,  5.67334552e-03],
       [-1.17262267e-02,  5.51567679e-02, -1.06701029e-03,
        -3.59460134e-04,  2.77549823e-02,  3.02205777e-03,
        -3.23932733e-02,  5.67334552e-03,  5.28601392e-02]]))

    Kernel    = NUTS
    HMC       = [NUTS(M, rng = rng[j], mass_matrix = mass_matrix, verbose = verbose, dt = 1e-5) for j in range(n_threads)]
    
    starting_point = np.array([np.float64(2.970836395983002), np.float64(2.1457700661243417), np.float64(-1.1216815578621249), np.float64(1126259462.4088995), np.float64(32.82289012101475), np.float64(0.8628497064389393), np.float64(-0.4819802030544022), np.float64(1.5720689487945567), np.float64(6.295442867400122)])
    
    samples = [H.sample(starting_point,
                          N=int(n_samps//n_threads),
                          position=j)
                 for j,H in enumerate(HMC)]
    
    import matplotlib.pyplot as plt
    from corner import corner
    corner(samples[0],
           labels=default_names,
           quantiles=[0.05, 0.5, 0.95], truths = None,
           show_titles=True, title_kwargs={"fontsize": 12}, smooth2d=1.0)
    
    plt.savefig("corner.pdf",bbox_inches='tight')
    
    fig = plt.figure()
    for i in range(len(default_names)):
        ax = fig.add_subplot(len(default_names),1,i+1)
        ax.plot(samples[0][:,i],'o-',markersize=2)
        ax.set_ylabel(default_names[i])
    plt.savefig("trace.pdf",bbox_inches='tight')
    

