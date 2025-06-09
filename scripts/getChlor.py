#*******************************************************************************
#                           ---------------
#                               getOC.py
#                           ---------------
#
# Description - Ocean color band ratio algorithms
#
# For further details, see:
# http://oceancolor.gsfc.nasa.gov/ANALYSIS/ocv6/
# http://oceancolor.gsfc.nasa.gov/ANALYSIS/kdv4/
# Jeremy Werdell, NASA Goddard Space Flight Center, 31 Jan 2012
#
# Translated from P.J Werdell's Matlab code to Python by:
# Lachlan McKinna, NASA GSFC
# July, 2014
#
# Zachary Erickson, NOAA PMEL
# June, 2025 - added PACE OCI algorithm
#
#*******************************************************************************
# Import python modules
import numpy as np;
#
#
# function to call empirical OC Chl and Kd algorithms
#
# both algorithms have the form:
# model = 10^(a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4)
# where a0 ... a4 are polynomial regression coefficients,
# x = log10(MBR), and MBR is maximum Rrs band ratio
#
# usage:
# chl = get_oc(Rrs443,Rrs490,Rrs510,Rrs555,type)
# where Rrs### is an array of Rrs at wavelength ###
# and type is 'OC4' or 'KD2S' or ... (see below)
#
# examples:
# for SeaWiFS Chl: oc4 = get_oc(Rrs443,Rrs490,Rrs510,Rrs555,'oc4')
# for MODIS Chl: oc3m = get_oc(Rrs443,Rrs488,-1,Rrs547,'oc3m')
# for SeaWiFS Kd: kd2s = get_oc(-1,Rrs490,-1,Rrs555,'kd2s')
#
# Note: this function expects four Rrs arrays in a particular order:
# blue (~443), blue (~488/490), green (~510/531), green (~547/555/560).
# Use a placeholder for algorithms that don't use four Rrs, such as -1
# in the above examples.  Some knowledge of what bands are used in each
# algorithm is therefore necessary; details provided here:
#
# http://oceancolor.gsfc.nasa.gov/ANALYSIS/ocv6/
# http://oceancolor.gsfc.nasa.gov/ANALYSIS/kdv4/
#
# Jeremy Werdell, NASA Goddard Space Flight Center, 31 Jan 2012
#
def getOC(**kwargs):
    #coefficients for the v6 (operational as of 2009) OC and KD algs
    #
    '''
    c = np.array(( (0.3272,-2.9940,2.7218,-1.2259,-0.5683),\ # OC4        1 SeaWiFS operational Chl
    (0.3255,-2.7677,2.4409,-1.1288,-0.4990),\   # OC4E       # MERIS operational Chl
    ( 0.3325,-2.8278,3.0939,-2.0917,-0.0257),\   # OC4O       # OCTS operational Chl
    ( 0.2515,-2.3798,1.5823,-0.6372,-0.5692),\   # OC3S       # SeaWiFS 3 band Chl
    ( 0.2424,-2.7423,1.8017, 0.0015,-1.2280),\   # OC3M       # MODIS operational Chl
    ( 0.2521,-2.2146,1.5193,-0.7702,-0.4291),\   # OC3E       # MERIS 3 band Chl
    ( 0.2399,-2.0825,1.6126,-1.0848,-0.2083),\   # OC3O       # OCTS 3 band Chl
    ( 0.3330,-4.3770,7.6267,-7.1457, 1.6673),\   # OC3C       # CZCS operational Chl
    ( 0.2511,-2.0853,1.5035,-3.1747, 0.3383),\   # OC2S       # SeaWiFS 2 band Chl
    ( 0.2389,-1.9369,1.7627,-3.0777,-0.1054),\   # OC2E      # MERIS 2 band Chl
    ( 0.2236,-1.8296,1.9094,-2.9481,-0.1718),\   # OC2O      # OCTS 2 band Chl
    ( 0.2500,-2.4752,1.4061,-2.8237, 0.5405),\   # OC2M      # MODIS 2 band Chl
    ( 0.1464,-1.7953,0.9718,-0.8319,-0.8073),\   # OC2M-HI   # MODIS high-res band Chl
    (-0.8515,-1.8263,1.8714,-2.4414,-1.0690),\   # KD2S      # SeaWiFS operational Kd
    (-0.8813,-2.0584,2.5878,-3.4885,-1.5061),\   # KD2M      # MODIS operational Kd
    (-0.8641,-1.6549,2.0112,-2.5174,-1.1035),\   # KD2E      # MERIS operational Kd
    (-0.8878,-1.5135,2.1459,-2.4943,-1.1043),\   # KD2O      # OCTS operational Kd
    (-1.1358,-2.1146,1.6474,-1.1428,-0.6190),\   # KD2C      # CZCS operational Kd
    ( 0.2228,-2.4683,1.5867,-0.4275,-0.7768),\   # OC3V      # VIIRS operational Chl
    ( 0.2230,-2.1807,1.4434,-3.1709, 0.5863),\   # OC2V      # VIIRS 2 band Chl
    (-0.8730,-1.8912,1.8021,-2.3865,-1.0453), \  # KD2V      # VIIRS operational Kd
    ( 0.32814,-3.20725,3.22969, -1.36769, -0.81739)); # PACE # PACE OCI Chl
    '''
    c = np.array(((0.3272,-2.9940,2.7218,-1.2259,-0.5683),\
        (0.3255,-2.7677,2.4409,-1.1288,-0.4990),\
        ( 0.3325,-2.8278,3.0939,-2.0917,-0.0257),\
        ( 0.2515,-2.3798,1.5823,-0.6372,-0.5692),\
        ( 0.2424,-2.7423,1.8017, 0.0015,-1.2280),\
        ( 0.2521,-2.2146,1.5193,-0.7702,-0.4291),\
        ( 0.2399,-2.0825,1.6126,-1.0848,-0.2083),\
        ( 0.3330,-4.3770,7.6267,-7.1457, 1.6673),\
        ( 0.2511,-2.0853,1.5035,-3.1747, 0.3383),\
        ( 0.2389,-1.9369,1.7627,-3.0777,-0.1054),\
        ( 0.2236,-1.8296,1.9094,-2.9481,-0.1718),\
        ( 0.2500,-2.4752,1.4061,-2.8237, 0.5405),\
        ( 0.1464,-1.7953,0.9718,-0.8319,-0.8073),\
        (-0.8515,-1.8263,1.8714,-2.4414,-1.0690),\
        (-0.8813,-2.0584,2.5878,-3.4885,-1.5061),\
        (-0.8641,-1.6549,2.0112,-2.5174,-1.1035),\
        (-0.8878,-1.5135,2.1459,-2.4943,-1.1043),\
        (-1.1358,-2.1146,1.6474,-1.1428,-0.6190),\
        ( 0.2228,-2.4683,1.5867,-0.4275,-0.7768),\
        ( 0.2230,-2.1807,1.4434,-3.1709, 0.5863),\
        (-0.8730,-1.8912,1.8021,-2.3865,-1.0453),\
        ( 0.32814, -3.20725, 3.22969, -1.36769, -0.81739)));
    #
    #slect coefficients and generate maximum band ratio
    #
    #switch lower(type)
    if 'type' not in kwargs:
        type = 'oc4';
    else:
        type = kwargs['type'];
        ##
    #
    if 'r1' not in kwargs:
        r1 = -999;
    else:
        r1 = kwargs['r1'];
        ##
    #
    if 'r2' not in kwargs:
        r2 = -999;
    else:
        r2 = kwargs['r2'];
        ##
    #
    if 'r3' not in kwargs:
        r3 = -999;
    else:
        r3 = kwargs['r3'];
        ##
    #
    if 'r4' not in kwargs:
        r4 = -999;
    else:
        r4 = kwargs['r4'];
        ##
    ##
    #
    if type == 'oc4e':
        a = c[1,:];
        r = np.log10(np.amax([r1,r2,r3]) / r4);
    if type == 'oc4o':
        a = c[2,:];
        r = np.log10(np.amax([r1,r2,r3]) / r4);
    if type ==  'oc3s':
        a = c[3,:];
        r = np.log10(np.amax([r1,r2]) / r4);
    if type == 'oc3m':
        a = c[4,:];
        r = np.log10(np.amax([r1,r2]) / r4);
    if type == 'oc3e':
        a = c[5,:];
        r = np.log10(np.amax([r1,r2]) / r4);
    if type == 'oc3o':
        a = c[6,:];
        r = np.log10(np.amax([r1,r2]) / r4);
    if type == 'oc3c':
        a = c[7,:];
        r = np.log10(np.amax([r1,r2])/ r4);
    if type == 'oc2s':
        a = c[8,:];
        r = log10(r2 / r4);
    if type == 'oc2e':
        a = c[9,:];
        r = np.log10(r2 / r4);
    if type == 'oc2o':
        a = c[10,:];
        r = np.log10(r2 / r4);
    if type == 'oc2m':
        a = c[11,:];
        r = np.log10(r2 / r4);
    if type == 'oc2m-hi':
        a = c[12,:];
        r = np.log10(r2 / r4);
    if type == 'kd2s':
        a = c[13,:];
        r = np.log10(r2 / r4);
    if type == 'kd2m':
        a = c[14,:];
        r = np.log10(r2 / r4);
    if type == 'kd2e':
        a = c[15,:];
        r = log10(r2 / r4);
    if type == 'kd2o':
        a = c[16,:];
        r = np.log10(r2 / r4);
    if type == 'kd2c':
        a = c[17,:];
        r = np.log10(r2 / r3);
    if type == 'oc3v':
        a = c[18,:];
        r = np.log10(np.amax([r1,r2]) / r4);
    if type == 'oc2v':
        a = c[19,:];
        r = np.log10(r2 / r4);
    if type == 'kd2v':
        a = c[20,:];
        r = np.log10(r2 / r4);
    if type == 'oc4':
        a = c[0,:];
        r = np.log10(np.amax([r1,r2,r3]) / r4);
    if type == 'pace':
        a = c[21,:];
        r = np.log10(np.amax([r1, r2, r3]) / r4);

    ##
    #calculate modeled parameter
    oc = np.copy(10.**(a[0] + a[1]*r + a[2]*r**2 + a[3]*r**3 + a[4]*r**4));
    #
    del a, r,c,kwargs;
    #
    return oc;
##
