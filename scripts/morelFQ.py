#*******************************************************************************
#                           ----------------
#                               morelFQ.py
#                           ----------------
#
# Description: functions used to return Morel F/Q air-surface interface 
#                parameters
#
# Lachlan McKinna
# NASA OBPG, July 2014
#
#*******************************************************************************
#--------------------
import os, sys,csv,gc;
from scipy import interpolate;
import numpy as np;
#
#--------------------
def morel_fq_appb(chl,solz):
    #
    #Get the morel fq_app coefficients
    morelFqAppbFile =  'common/morel_fq_appb.txt';
    morelFqAppbData = np.loadtxt(morelFqAppbFile,unpack=True);
    #
    #Extract coefficients from data block
    f0 = morelFqAppbData[0: 6,:];
    sf = morelFqAppbData[6:12,:];
    q0 = morelFqAppbData[12:18,:];
    sq = morelFqAppbData[18:24,:];
    #
    wlut = np.array((412.5, 442.5, 490., 510., 560., 620., 660.));
    clut = np.array((0.03, 0.1, 0.3, 1., 3., 10.));
    wx,cx = np.meshgrid(wlut,clut);
    #
    z = 1.0 - np.cos(solz * np.pi / 180.);
    #
    f1 = f0 + sf * z;
    q1 = q0 + sq * z;
    #
    if chl > clut[5]:
        chl = clut[5];
    if chl < clut[0]:
        chl = clut[0];
    #
    if chl in clut:
        indx = np.where(clut==chl)[0][0];
        fOut = f1[indx];
        qOut = q1[indx];
        #
    elif chl not in clut:
        clutNew = np.append(clut,chl);
        clutNew = np.sort(clut);
        print clutNew;
        #
        indx = np.where(clutNew == chl)[0][0];
        #
        fInterp = interpolate.interp2d(wx, cx, f1, kind='linear');
        qInterp = interpolate.interp2d(wx, cx, q1, kind='linear');
        #
        f1 = fInterp(wlut,clutNew);
        q1 = qInterp(wlut,clutNew);
        #
        fOut = f1[indx];
        qOut = q1[indx];
        ##
    #
    return fOut,qOut;
##
#----
#----
#Function readFQ - loads data from LUT file and sorts into multi-dimensional array
def readFq():
    #
    #dimensions
    nw = 7; # wavelengths
    ns = 6; # zenith
    nc = 6; # chl
    nn = 17; # nadir
    na = 13; # azimuth

    #Get the morel fq coefficients
    morelFqFile = "common/morel_fq.dat";
    morelFqData = np.loadtxt(morelFqFile);

    foq = np.zeros((nw,ns,nc,nn,na));
    count = 0;
    #
    for i in range(0,nw):
        for j in range(0,ns):
            for k in range(0,nc):
                for l in range(0,nn):
                    foq[i,j,k,l,:] = morelFqData[count,:];
                    count = count + 1;
                    ##
                ##
            ##
        ##
    ##
    return foq;
##
#
def get_fq(w,s,chl_in,n,a,foq):
    #
    fqint = -999;
    #
    wvl = np.array((412.5,442.5,490.,510.,560.,620.,660.));
    sun = np.array((0,15,30,45,60,75));
    chl = np.array((0.03,0.1,0.3,1,3,10));
    nad = np.array((1.078,3.411,6.289,9.278,12.3,15.33,18.37,21.41,24.45,27.5,30.54,33.59,36.64,39.69,42.73,45.78,48.83));    
    azm = np.array((0,15,30,45,60,75,90,105,120,135,150,165,180));
    #
    nw = 6;
    ns = 5;
    nc = 5;
    nn = 16;
    na = 12;
    #
    lchl = np.log(chl);
    c = np.log(chl_in);
    #
    #locate nearest wavelength
    idx = np.where(np.fabs(wvl - w) < 15);
    #I removed error checking here...
    iw = idx[0][0];
    #print iw
    #print w
        ##
    ##
    #locate boundary zenith
    if s <= sun[0]:
        js = 0;
    elif s >= sun[ns]:
        js = ns;
    else:
        js = 0;
        while s > sun[js + 1]:
            js = js + 1;
            ##
        ##
    ##
    #
    #locate boundary chl
    if c <= lchl[0]:
        kc = 0;
    elif c >= lchl[nc]:
        kc = nc;
    else:
        kc = 0;
        while c > lchl[kc + 1]: 
            kc = kc + 1;
            ##
        ##
    ##
    #
    #locate boundary nadir 
    if n <= nad[0]: 
        n = nad[0]; # force min nadir angle to min table value
        ln = 0;
    elif n >= nad[nn]:
        ln = nn;
    else:
        ln = 0;
        while n > nad[ln + 1]: 
            ln = ln + 1;
            ##
        ##
    ##
    #locate boundary azimuth
    if a <= azm[0]:
        ma = 1;
    elif a >= azm[na]:
        ma = na;
    else:
        ma = 1;
        while a > azm[ma + 1]:
            ma = ma + 1;
            ##
        ##
    ##
    #weight, interpolate, and iterate values to solve for f/Q
    #
    ds = np.array((-999,-999));
    dc = np.array((-999,-999));
    dn = np.array((-999,-999));
    da = np.array((-999,-999));
    #
    ds[0] = (sun[js+1] - s) / (sun[js+1] - sun[js]);
    ds[1] = (s - sun[js]) / (sun[js+1] - sun[js]);
    #
    dc[0] = (lchl[kc+1] - c) / (lchl[kc+1] - lchl[kc]);
    dc[1] = (c - lchl[kc]) / (lchl[kc+1] - lchl[kc]);
    #
    dn[0] = (nad[ln+1] - n) / (nad[ln+1] - nad[ln]);
    dn[1] = (n - nad[ln]) / (nad[ln+1] - nad[ln]);
    #
    da[0] = (azm[ma+1] - a) / (azm[ma+1] - azm[ma]);
    da[1] = (a - azm[ma]) / (azm[ma+1] - azm[ma]);
    #
    print ds,dc,dn,da
    #
    fqint = 0;
    #
    for j in range(0,2):
        for k in range(0,2):
            for l in range(0,2):
                for m in range(0,2):
                    fqint = fqint + ds[j] * dc[k] * dn[l] * da[m] * foq[iw, js+j-1, kc+k-1, ln+l-1, ma+m-1];
                    ##
                ##
            ##
        ##
    ##
    return fqint;
#
#
def morel_fq(chl,solz,theta,relaz,**kwargs):
    #
    #
    fqa = -999;
    fqc = -999;
    #
    #Load foq data - cll the readFq function
    foq_data = readFq();
    #
    h2o = 1.34;
    w = np.array((412.5,442.5,490,510,560,620,660));
    thetap = np.arcsin(np.sin(theta * (np.pi/180.0)) / h2o) / (np.pi/180.0);
    #
    lenlam = len(w);
    fq = np.zeros((lenlam,));
    f0 = np.zeros((lenlam,));
    #
    for i in range(0,7):
        fq[i] = get_fq(w[i],solz,chl,thetap,relaz,foq_data);
        f0[i] = get_fq(w[i],0,chl,0,0,foq_data);
        ##
    ##
    #
    print f0
    print fq
    fc = f0 / fq;
    #
    if 'lam' not in kwargs:
        lam = np.arange(380,701,1);
    else: 
        lam = kwargs['lam'];
    #
    fqInterp = interpolate.interp1d(w,fq,bounds_error=False,kind='cubic',fill_value=0.0);
    fcInterp = interpolate.interp1d(w,fc,bounds_error=False,kind='cubic',fill_value=0.0);
    #
    fqOut= fqInterp(lam);
    fcOut = fcInterp(lam);
    ##
    return fqOut, fcOut;
##