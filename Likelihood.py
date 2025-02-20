

from astropy import constants as const
M_sun = const.M_sun.value
G = const.G.value
c = const.c.value
pc = const.pc.value
import jax.numpy as np

from utils import GreenwichMeanSiderealTime

def TaylorF2(Mc, q, phi_c, distance, iota, frequency_array):
        
    nu = q/((1+q)**2)
    f_max = frequency_array[-1]

    r = distance
    #print(pc)
    Mc = Mc*M_sun
    r = r*pc*1e6#megaparsec
    M = Mc/(nu**(3/5))

    f_lso =f_max/2
    #nu = m1*m2/(m1+m2)**2
    v = (G*np.pi*M*frequency_array)**(1/3)
    #print(v)
    v = v/c
    
    v_lso = (G*np.pi*M*f_lso)**(1/3)
    v_lso = v_lso /c
    
    
    gamma  = np.euler_gamma

    
    amp =(np.pi**(-2/3))*np.sqrt(5/24)*(G*Mc/(c**3))**(5/6)*frequency_array**(-7/6)*(c/r)###questo è ok
    phi_plus=  ((3)/(128*nu*(v)**5))*(1
    
    +
    #0PN
    v**2*(20/9)*(743/336 + nu*11/4)
    -\
    #1PN
    16*np.pi*v**3+ \
    #1.5PN
    10*(v**4)*(3058673/1016064 +nu*5429/1008+(nu**2)*617/144)
    +\
    #2PN
    v**(5)*np.pi*(38645/756-nu*65/9)*(1+3*np.log(v))
    +\
    #2.5PN
    (v**6)*(11583231236531/4694215680 -np.pi**2*640/3 -6848*gamma/21 - 6848/21*np.log(4*v)+\

    nu*(-15737765635/3048192+ 2255*(np.pi**2)/12)+nu**2*76055/1728 -nu**3*127825/1296)+\
    #3PN
    v**(7)*np.pi*(77096675/254016 +nu*378515/1512 -(nu**2)*74045/756))


    phi_plus += np.pi- np.pi/4


    phi_cross = phi_plus  + np.pi/2
    #phi= np.exp(1j*(-(phi_c) + 2*np.pi*frequency_array*t_c ))
    phi= np.exp(1j*(-(phi_c)))
    h_plus = phi*amp*((1+(np.cos(iota))**2)/2)*np.exp(1j*phi_plus)
    h_cross = phi*amp*np.cos(iota)*np.exp(1j*phi_cross)
    '''
    f_tot =np.arange(0, f_max, delta_f)
    h_pad = np.zeros(len(f_tot)-len(f))
    h_plus = np.concatenate((h_pad, h_plus))
    h_cross = np.concatenate((h_pad, h_cross))
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

    def __init__(self, name, t_ref=0):
        """Constructor"""

        self.name = name
        self.t_ref = t_ref

        if name not in self.available_detectors.keys():
            raise ValueError("Not valid argument ({}) for 'name' parameter.".format(name))

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





        a_ = (1/16)*np.sin(2*g_)*(3-np.cos(2*lat))*(3-np.cos(2*dec))*np.cos(2*(ra - lst))-\
             (1/4)*np.cos(2*g_)*np.sin(lat)*(3-np.cos(2*dec))*np.sin(2*(ra - lst))+\
             (1/4)*np.sin(2*g_)*np.sin(2*lat)*np.sin(2*dec)*np.cos(ra - lst)-\
             (1/2)*np.cos(2*g_)*np.cos(lat)*np.sin(2*dec)*np.sin(ra - lst)+\
             (3/4)*np.sin(2*g_)*(np.cos(lat)**2)*(np.cos(dec)**2)

        b_ = np.cos(2*g_)*np.sin(lat)*np.sin(dec)*np.cos(2*(ra - lst))+\
             (1/4)*np.sin(2*g_)*(3-np.cos(2*lat))*np.sin(dec)*np.sin(2*(ra - lst))+\
             np.cos(2*g_)*np.cos(lat)*np.cos(dec)*np.cos(ra - lst)+\
             (1/2)*np.sin(2*g_)*np.sin(2*lat)*np.cos(dec)*np.sin(ra - lst)


        return a_, b_
    


    def project_waveform(self, freq_array, Mc, q, phi_c, distance, iota, right_ascension, declination, polarization, GPS_time):

        t_c = GPS_time
        h_plus, h_cross = TaylorF2(Mc, q, phi_c,  distance, iota, freq_array)
        #gmst = np.radians(self.lst_estimate(GPS_time))
        fplus, fcross = self.antenna_pattern_functions(right_ascension, declination, polarization, GPS_time)
        h = fplus*h_plus + fcross*h_cross
        return h 



  



    def antenna_pattern_functions(self, right_ascension, declination, polarization, gps_time):
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

        ra = np.radians(right_ascension)
        dec = np.radians(declination)

        pol = np.radians(polarization)
        lat = np.radians(self.latitude)
        g_ = np.radians(self.gamma)
        z_ = np.radians(self.zeta)
        gmst = np.mod(GreenwichMeanSiderealTime(gps_time), 2*np.pi)
        lst = gmst + np.radians(self.longitude)
        ampl11, ampl12 = self._ab_factors(g_, lat, ra, dec, lst)

        
        fplus = np.sin(z_)*(ampl11*np.cos(2*pol) + ampl12*np.sin(2*pol))
        fcross = np.sin(z_)*(ampl12*np.cos(2*pol) - ampl11*np.sin(2*pol))


        return fplus, fcross
    

















class Likelihood():
    def __init__(self, ):
        self.detector = GWDetector('H1')
        


    def project_waveform(self, freq_array, Mc, q, phi_c, distance, iota, right_ascension, declination, polarization, GPS_time):
        return np.real(np.array(self.detector.project_waveform(freq_array, Mc, q, phi_c, distance, iota, right_ascension, declination, polarization, GPS_time)))
        




    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from jax import jacrev
    freq_array = np.arange(10, 100, 0.1)
    import jax
    jax.config.update("jax_enable_x64", True)
    like = Likelihood(freq_array)
    projected_waveform = like.project_waveform

    #jacrev(function to differentiate,[arguments wrt derive position])
    
    gradient_function = jacrev(projected_waveform, 9)
    

    #always 
    plt.plot(freq_array, (gradient_function(freq_array, 30., 0.5, 0., 1000., 0., 0., 0., 0., 1224010763. )))



    plt.show()
    

