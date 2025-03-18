from astropy import constants as const
M_sun = const.M_sun.value
G = const.G.value
c = const.c.value
pc = const.pc.value
import jax.numpy as jnp
import numpy as np
from functools import partial
import jax
from utils import GreenwichMeanSiderealTime
from utils import TimeDelayFromEarthCenter, Masses2McQ
from granite.powerspectrum.mesa import psd_onsource
from granite.noise.noise import load_data, generate_data

import sys

def log_prior(params):
#    default_names = ['phiref','ra','dec','tc','mc','q','costheta_jn','psi','logdistance']
 
    logP = 0.0
#    logP += 3.0*params[8]
#
#    # declination
#    logP += jnp.log(jnp.abs(jnp.cos(params[2])))

    # chirp mass and mass ratio
#    mc      = params[4]
#    q       = params[5]
    mc      = params[0]
    q       = params[1]
    logP   += jnp.log(mc)
    logP   += (2./5.)*jnp.log(1.0+q)-(6./5.)*jnp.log(q)
    return logP

def in_bounds(param, bounds):
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
    return all(bounds[n][0] < param[n] < bounds[n][1] for n in param.keys())

def log_posterior(params, detector_list):
    
    return  log_prior(params) + log_likelihood(params, detector_list)

#        @partial(jax.jit, static_argnums = (0))
def log_likelihood(params, detector_list):
    # Ensure the list of log-likelihoods is a JAX array
    log_likelihoods = jnp.array([single_detector_log_likelihood(params, det) for det in detector_list])

    # Then use jnp.sum
    return jnp.sum(log_likelihoods)

def single_detector_log_likelihood(params, detector_dictionary):
    h = project_waveform(params, detector_dictionary)
    residuals = detector_dictionary["FrequencySeries"] - h
#    jax.debug.print("sante!")
#    import matplotlib.pyplot as plt
#    plt.plot(detector_dictionary["Frequency"],h,label="h")
#    plt.plot(detector_dictionary["Frequency"],residuals,label="res")
#    plt.plot(detector_dictionary["Frequency"],detector_dictionary["FrequencySeries"],label="data")
#    plt.legend()
#    plt.show()
#    exit()
    return -detector_dictionary["TwoDeltaTOverN"]*jnp.vdot(residuals, residuals/detector_dictionary["sigmasq"]).real
    
def project_waveform(params, detector_dictionary):
        #    default_names = ['mc','q','phiref','ra','dec','tc','costheta_jn','psi','logdistance']
    
    f = detector_dictionary["Frequency"]
    h_plus, h_cross = TaylorF2(params, f)
    #gmst = np.radians(self.lst_estimate(GPS_time))
    latitute  = detector_dictionary["latitude"]
    longitude = detector_dictionary["longitude"]
    gamma     = detector_dictionary["gamma"]
    zeta      = detector_dictionary["zeta"]
    
    fplus, fcross   = antenna_pattern_functions(params, latitute, longitude, gamma, zeta)
    
    ra = np.float64(2.1457700661243417)#np.radians(right_ascension)
    dec = np.float64(-1.1216815578621249)#np.radians(declination)
    tc  = np.float64(1126259462.4088995)

    timedelay       = TimeDelayFromEarthCenter(latitute, longitude, ra, dec, tc)
    timeshift       = timedelay
    shift           = 2.0*np.pi*f*timeshift

    h = (fplus*h_plus + fcross*h_cross)*(jnp.cos(shift)-1j*jnp.sin(shift))
    return h

#@partial(jax.jit, static_argnums=(1,2,3,4))
def antenna_pattern_functions(params, det_latitute, det_longitude, det_gamma, det_zeta):
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
#        ra = np.float64(2.1457700661243417)
#        dec =  np.float64(-1.1216815578621249)
#        pol = np.float64(1.5720689487945567)
#        tc = np.float64(1126259462.423)



    ra = np.float64(2.1457700661243417)#np.radians(right_ascension)
    dec = np.float64(-1.1216815578621249)#np.radians(declination)

    pol = np.float64(1.5720689487945567)#np.radians(polarization)
    tc  = np.float64(1126259462.4088995)
    lat = jnp.radians(det_latitute)
    g_ = jnp.radians(det_gamma)
    z_ = jnp.radians(det_zeta)
    gmst = jnp.mod(GreenwichMeanSiderealTime(tc), 2*jnp.pi)
    lst = gmst + jnp.radians(det_longitude)
    ampl11, ampl12 = _ab_factors(g_, lat, ra, dec, lst)

    c2pol = jnp.cos(2*pol)
    s2pol = jnp.sin(2*pol)
    
    fplus = jnp.sin(z_)*(ampl11*c2pol + ampl12*s2pol)
    fcross = jnp.sin(z_)*(ampl12*c2pol - ampl11*s2pol)

    return fplus, fcross

@jax.jit
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
    s2g = jnp.sin(2*g_)
    c2g = jnp.cos(2*g_)
    cdec  = jnp.cos(dec)
    sdec  = jnp.sin(dec)
    c2dec = jnp.cos(2*dec)
    s2dec = jnp.sin(2*dec)
    clat  = jnp.cos(lat)
    slat  = jnp.sin(lat)
    c2lat = jnp.cos(2*lat)
    s2lat = jnp.sin(2*lat)
    
    a_ = (1/16)*s2g*(3-c2lat)*(3-c2dec)*jnp.cos(2*(ra - lst))-\
         (1/4)*c2g*slat*(3-c2dec)*jnp.sin(2*(ra - lst))+\
         (1/4)*s2g*s2lat*s2dec*jnp.cos(ra - lst)-\
         (1/2)*c2g*clat*s2dec*jnp.sin(ra - lst)+\
         (3/4)*s2g*(clat**2)*(cdec**2)

    b_ = c2g*slat*sdec*jnp.cos(2*(ra - lst))+\
         (1/4)*s2g*(3-c2lat)*sdec*jnp.sin(2*(ra - lst))+\
                 c2g*clat*cdec*jnp.cos(ra - lst)+\
         (1/2)*s2g*s2lat*cdec*jnp.sin(ra - lst)


    return a_, b_

def TaylorF2(params, frequency_array):
    # Extract parameters

      
    Mc, q, phi_c, logdistance, costheta_jn = params[0], params[1], np.float64(2.970836395983002),np.float64(6.505442867400122), np.float64(-0.4819802030544022)
#    Mc, q = params[0], params[1],
#    phi_c = np.float64(2.970836395983002)
#    logdistance = np.float64(6.295442867400122)
#    costheta_jn = np.float64(-0.4819802030544022)

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
    

    def __init__(self,
                 name,
                 datafile           = None,
                 psd_file           = 'LIGO-P1200087-v18-aLIGO_DESIGN_psd.dat',
                 simulation         = False,
                 psd_method         = 'mesa-on-source',
                 T                  = 2.0,
                 starttime          = 1126259461.423,
                 trigtime           = 1126259462.423,
                 sampling_rate      = 1024.,
                 flow               = 20,
                 fhigh              = None,
                 zero_noise         = True,
                 calibration        = None,
                 download_data      = 1,
                 datalen_download   = 32,
                 channel            = '',
                 gwpy_tag           = None):
                 
        # initialise the needed attributes
        self.name             = name
        self.latitude         = available_detectors[name][0]
        self.longitude        = available_detectors[name][1]
    
        if name not in available_detectors.keys():
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

        # set the maximum frequency cutoff to prevent aliasing
        if fhigh is None:
            self.fhigh = self.sampling_rate*0.45
            sys.stdout.write('\nMaximum frequency not given: it will be set to sampling_rate*0.45 to prevent aliasing\n')
        elif fhigh>self.sampling_rate/2.:
            self.fhigh = self.sampling_rate*0.45
            sys.stdout.write('\nMaximum frequency above the Nyquist bound: it will be set to sampling_rate*0.45 to prevent aliasing\n')
        else:
            self.fhigh = fhigh

        if self.channel is not None:
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
        else:
            self.Times, self.TimeSeries, self.Frequency, self.FrequencySeries, self.PowerSpectralDensity, self.mesa_object = generate_data(self.psd_file,
                                                            T = self.T,
                                                            starttime     = self.Epoch,
                                                            sampling_rate = self.sampling_rate,
                                                            fmin          = self.flow,
                                                            fmax          = self.fhigh,
                                                            zero_noise    = self.zero_noise,
                                                            asd           = False)

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

        self.latitude = available_detectors[name][0]
        self.longitude = available_detectors[name][1]
        self.gamma = available_detectors[name][2]
        self.zeta = available_detectors[name][3]

def check_antenna_pattern(name):
    params = np.array([
                       np.float64(2.970836395983002),
                       np.float64(2.1457700661243417),
                       np.float64(-1.1216815578621249),
                       np.float64(1126259462.4088995),
                       np.float64(32.82289012101475),
                       np.float64(0.8628497064389393),
                       np.float64(-0.4819802030544022),
                       np.float64(1.5720689487945567),
                       np.float64(6.295442867400122)
                       ])

    latitude = available_detectors[name][0]
    longitude = available_detectors[name][1]
    gamma = available_detectors[name][2]
    zeta = available_detectors[name][3]
    
    x = np.linspace(0,2*np.pi, 100)
    y = np.linspace(-np.pi/2.,np.pi/2., 100)
    Z = np.zeros((x.shape[0],y.shape[0]))
    
    from tqdm import tqdm
    
    for i in tqdm(range(x.shape[0])):
        for j in range(y.shape[0]):
            params[1] = x[i]
            params[2] = y[j]
            fp, fc = antenna_pattern_functions(params, latitude, longitude, gamma, zeta)
                        
            Z[i,j] = (fp**2+fc**2)**(0.5)

    X, Y = np.meshgrid(x, y)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    C = ax.contourf(X, Y, Z.T, 100)
    fig.colorbar(C)
    plt.show()
    return

def check_mass_prior():

    from hmc_func import compute_mass_matrix

    x = np.linspace(5.0,40.0, 100)
    y = np.linspace(0.125,1.0, 100)
    Z = np.zeros((x.shape[0],y.shape[0]))
    
    detectors = detector_constructor(["H1"], channel =None)
    logP = jax.jit(log_prior)
    H = jax.hessian(log_prior)
    
    from tqdm import tqdm
    
    for i in tqdm(range(x.shape[0])):
        for j in range(y.shape[0]):
            params = np.hstack((x[i],y[j]))
            Z[i,j] = logP(params)
#            print("{} {} mc = {} q = {} H = {} invM = {}".format(i,j,x[i],y[j], np.linalg.inv(H(params)), compute_mass_matrix(H, params)[1]))

    X, Y = np.meshgrid(x, y)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    C = ax.contour(X, Y, Z.T, 100)
    fig.colorbar(C)
    plt.show()
    return

def check_likelihood(detector_dictionary):

    from hmc_func import compute_mass_matrix

    x = np.linspace(5.0,40.0, 100)
    y = np.linspace(0.125,1.0, 100)
    Z = np.zeros((x.shape[0],y.shape[0]))
    
    logL = jax.jit(partial(log_likelihood, detector_list=[detector_dictionary]))
    H = jax.hessian(logL)
    
    from tqdm import tqdm
    
    for i in tqdm(range(x.shape[0])):
        for j in range(y.shape[0]):
            params = np.hstack((x[i],y[j]))
            Z[i,j] = logL(params)
            print("{} {} mc = {} q = {} H = {} invM = {}".format(i,j,x[i],y[j], np.linalg.inv(H(params)), compute_mass_matrix(H, params)[1]))

    X, Y = np.meshgrid(x, y)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    C = ax.contour(X, Y, Z.T, 100)
    fig.colorbar(C)
    plt.show()
    return

def check_posterior(detector_dictionary):

    from hmc_func import compute_mass_matrix

    x = np.linspace(5.0,40.0, 256)
    y = np.linspace(0.125,1.0, 256)
    Z = np.zeros((x.shape[0],y.shape[0]))
    
    logP = jax.jit(partial(log_posterior, detector_list=[detector_dictionary]))
    H = jax.hessian(logP)
    
    from tqdm import tqdm
    
    for i in tqdm(range(x.shape[0])):
        for j in range(y.shape[0]):
            params = np.hstack((x[i],y[j]))
            Z[i,j] = logP(params)
#            print("{} {} mc = {} q = {} H = {} invM = {}".format(i,j,x[i],y[j], np.linalg.inv(H(params)), compute_mass_matrix(H, params)[1]))

    X, Y = np.meshgrid(x, y)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    C = ax.contour(X, Y, Z.T, 100)
    fig.colorbar(C)
    plt.show()
    return

def check_waveform_projection(name):
    params = np.array([
                       np.float64(2.970836395983002),
                       np.float64(2.1457700661243417),
                       np.float64(-1.1216815578621249),
                       np.float64(1126259462.4088995),
                       np.float64(2.82289012101475),
                       np.float64(0.8628497064389393),
                       np.float64(-0.4819802030544022),
                       np.float64(1.5720689487945567),
                       np.float64(6.295442867400122)
                       ])
    detector_dictionary = {"latitude":available_detectors[name][0],
                           "longitude":available_detectors[name][1],
                           "gamma":available_detectors[name][2],
                           "zeta":available_detectors[name][3]}
    frequency_array     = np.linspace(20,512,10000)
    h = project_waveform(params, {"Frequency":frequency_array})
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.plot(frequency_array, h)
    plt.show()
    return

def detector_constructor(names, channel = None):
    
    if channel == None:
        print("we are simulating data")
    elif channel == "GWOSC":
        print("we are downloading public data")
    else:
        print("we do not really know what to do yet")
        
    
    D = [GWDetector(name, channel = channel).__dict__ for name in names]
    return D

def inject_signal_in_noise(params,
                           detector_dictionary):

    h = project_waveform(params, detector_dictionary)
    
    # add to the detector noise
    detector_dictionary["FrequencySeries"] += h
    
    # signal-to-noise ratio
    SNR = np.sqrt(4.0*detector_dictionary["df"]*jnp.sum(jnp.conj(h)*h/detector_dictionary["PowerSpectralDensity"]).real)
    
    print('\nInjected SNR = %.2f' %(SNR))
    
    return SNR, h



if __name__=="__main__":
    available_detectors = {
        'V1': [43.63, 10.5, 115.56, 90.],
        'H1': [46.45, -119.41, 170.9, 90.],
        'L1': [30.56, -90.77, 242.7, 90.],
        'GEO600': [52.25, -9.81, 68.775, 94.33],
        'TAMA300': [35.68, -139.54, 225., 90.],
        'ET': [40.44, 9.4566, 116.5, 60.], # Sardinia site hypothesis
        'K': [36.41, 137.30, 15.36, 90.]
    }

    detectors = detector_constructor(["H1"], channel =None)
    
    q_inj = np.array([
                        np.float64(18.2289012101475),
                       np.float64(0.628497064389393),
                       np.float64(2.970836395983002),
                       np.float64(2.1457700661243417),
                       np.float64(-1.1216815578621249),
                       np.float64(1126259462.4088995),
                       np.float64(-0.4819802030544022),
                       np.float64(1.5720689487945567),
                       np.float64(6.505442867400122)
                       ])
    
    q0 = np.array([
                       np.float64(15.0289012101475),
                       np.float64(0.97064389393)
                       ])

#    q0 = q_inj[:2]
    import matplotlib.pyplot as plt
#    plt.plot(detectors[0]["Frequency"], detectors[0]["FrequencySeries"])
    snr, h_inj = inject_signal_in_noise(q_inj, detectors[0])
#    
#    plt.plot(detectors[0]["Frequency"], h_inj)
#    plt.show()
#    exit()
#    check_posterior(detectors[0])
#    exit()
    logp = jax.jit(partial(log_posterior, detector_list = detectors))#jax.jit()

    from hmc_func import test_integrator, compute_mass_matrix
    
#    print(logp(q0))
    rng = np.random.default_rng(seed = 232)
    n_steps = 10000
    n_leaps = 500
    step_size = 0.005
    
    _, inverse_metric_0, _ = compute_mass_matrix(jax.hessian(logp),q0)
    p0 = np.dot(np.linalg.cholesky(inverse_metric_0).T,rng.normal(size=q0.shape[0]))
    test_integrator(p0, q0, n_leaps, step_size, logp, inverse_metric_0)
    exit()
    
    from hmc_func import run_nuts_rmhmc, run_rmhmc
    
    qs = run_rmhmc(q0, n_steps, n_leaps, step_size, logp, rng)

    thinning = int(max([acl(q) for q in qs.T]))
    print("ACL = {}".format(thinning))
    qs = qs[::thinning]

    x = np.linspace(10,50,200)
    y = np.linspace(0.1,1.0,200)
    Z = np.array([logp(np.array([xi,yi])) for yi in y for xi in x]).reshape(x.shape[0],y.shape[0])

    X, Y = np.meshgrid(x,y)
    
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.axvline(q_inj[0], color='r')
    ax.axhline(q_inj[1], color='r')
    C = ax.contour(X, Y, Z, 32)
    ax.plot(qs[:,0],qs[:,1],'o-',alpha=0.5,lw=0.3)
    fig.colorbar(C)
    fig.savefig("likelihood.png")
    plt.show()
