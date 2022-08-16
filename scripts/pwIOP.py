#*******************************************************************************
#                           ----------------
#                               pwIOP.py
#                           ----------------
#
# Description: functions used to return pure water absorption and backscattering
#              coefficients.
#
# Lachlan McKinna
# NASA OBPG, July 2014
#
#*******************************************************************************
#Import modules
import os,sys;
import numpy as np;
from scipy import interpolate;
#-------
#--------------
#get the pure water bacscattering coefficient
def get_bbw(wl):
    bbw = np.copy(0.0038 * (400.0 / wl)**4.32);
    del wl;
    return bbw;
#--------------
#--------------
def datainterp(commonwav,wavin,datain):
    #
    data = np.interp(commonwav, wavin, datain, left=None, right=None);
    #
    return data;
#--------------
#--------------
#Get the pure water absorption coefficient and interpolate to sensor wavelength
def get_aw(wl):
    #
    pwFile = os.path.join(os.path.dirname(__file__), "common/pw_optics_coef.txt");
    #print (os.getcwd(".."))
    #pwData = np.loadtxt(pwFile);
    #
    fid=open(pwFile);
    lines=fid.readlines();
    #
    nrows = len(lines);
    ncols = len(lines[0].split());
    #
    pwData = np.zeros((nrows,ncols));
    #
    idx=0;
    for line in lines:
        pwLine = np.asarray(line.split(),dtype=np.float);
        pwData[idx,:] = np.copy(pwLine);
        del pwLine
        idx+=1;
        ##
    fid.close();
    #
    aw = datainterp(wl,pwData[:,0],pwData[:,1]);
    #
    del lines, pwData,nrows,ncols,fid;
    #
    return aw;
##
#*******************************************************************************
