import jax.numpy as np
import jax
from math import pi, floor
jax.config.update("jax_enable_x64", True)



# Constants
JULIAN_DATE_START_OF_GPS_TIME = 2444244.5
leaps = np.array([
    46828800, 78364801, 109900802, 173059203, 252028804, 315187205, 346723206, 
    393984007, 425520008, 457056009, 504489610, 551750411, 599184012, 820108813, 
    914803214, 1025136015, 1119744016, 1167264017
], dtype=np.float64)

EPOCH_J2000_0_GPS = 630763213

def GreenwichMeanSiderealTime(gpstime) :
    """Calculates Greenwich Mean Sidereal Time given GPS time."""
    return _GreenwichMeanSiderealTime(gpstime)

def _GreenwichMeanSiderealTime(gpstime):
    jd = _GPS2JD(gpstime)
    gps_ns = gpstime - np.round(gpstime)
    t_hi = (jd - 2451545.0) / 36525.0
    t_lo = gps_ns / (36525.0 * 86400.0)
    t = t_hi + t_lo

    sidereal_time = (-6.2e-6 * t + 0.093104) * t * t + 67310.54841
    sidereal_time += 8640184.812866 * t_lo
    sidereal_time += 3155760000.0 * t_lo
    sidereal_time += 8640184.812866 * t_hi
    sidereal_time += 3155760000.0 * t_hi

    return sidereal_time * pi / 43200.0

def GPS2JD(gpstime):
    """Converts GPS time to Julian Date."""
    return _GPS2JD(gpstime)

def _GPS2JD(gpstime):
    """Helper function to compute Julian Date from GPS time."""
    dot2gps = 29224.0
    dot2utc = 2415020.5
    
    # Determine leap seconds
    if gpstime < 820108814:
        nleap = 32
    elif 820108814 <= gpstime < 914803215:
        nleap = 33
    else:
        nleap = 34

    dot = dot2gps + (gpstime - (nleap - 19)) / 86400.0
    utc = dot + dot2utc
    jd = utc

    return jd



def TimeDelayFromEarthCenter( lat, lon,  ra,dec,GPS_time,):
  
    # Constants
    
    lat = np.radians(lat)
    lon = np.radians(lon)
    EarthRadius = 6371.0*1e3
    c  = 2.99792458*1e8
    lst = GreenwichMeanSiderealTime(GPS_time) + lon 
    
    dec = np.radians(90- dec)
    ra = np.radians(ra)
    s = np.array([np.sin(dec)*np.cos(ra), np.sin(dec)*np.sin(ra), np.cos(dec)])
    d = np.array([np.cos(lat)*np.cos(lst), np.cos(lat)*np.sin(lst), np.sin(lat)])
    deltaT = -(np.dot(s, d))*EarthRadius/c
    return deltaT








if __name__ == '__main__':
    from astropy.time import Time
    time = Time(1224010763, format='gps', scale='utc',
                        location=(0,0)).sidereal_time('mean').rad
    print(time)
    import numpy as np
    # Example: Compute GMST for total GPS seconds 1360800000
    gps_seconds = 1224010763
    GMST = np.mod((GreenwichMeanSiderealTime(gps_seconds)), 2*np.pi)
    print(f"GMST: {GMST:.6f} degrees")
