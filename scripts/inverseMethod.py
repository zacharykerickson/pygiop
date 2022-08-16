#*******************************************************************************
#                           ----------------
#                               invert.py
#                           ----------------
#
# Description: define inverse solution methods:
              #lm - Levenberg-Marquardt
              #lmi - linear matrix inversion
              #amoeba - amoeba (nelder-mead) simplex optimization
#
# Lachlan McKinna
# NASA OBPG, July 2014
#
#*******************************************************************************
#Import necessary modules
import os,sys;
from scipy.optimize import leastsq;
from scipy.optimize import least_squares;
from scipy.optimize import minimize;
from . import mpfit;
from . import ancIOP
import numpy as np;
import matplotlib.pyplot as plt;
import time;
#
#-----------------
#****FOR levmar and scipy(least_squares) Levenberg-Marquart implementaton*****
#Define forward reflectance model here
#-----------------
def fitModel(p,lam,aw,bbw,aphStar,sdg,eta,g0,g1):
    if (len(p)==3):
        adgStar = ancIOP.get_adgStar(lam,sdg);
        bbpStar = ancIOP.get_bbpStar(lam,eta);
    elif len(p)==5:
        adgStar = ancIOP.get_adgStar(lam,p[3]);
        bbpStar = ancIOP.get_bbpStar(lam,p[4]);
    atot = aw + p[0]*aphStar + p[1]*adgStar;
    bbtot = bbw + p[2]*bbpStar;
    kappa = atot + bbtot;
    u = bbtot / kappa;
    rrsMod = np.copy((g0 + g1*u)*u);
    #
    return rrsMod;
#

#-----------------
#****FOR MPFIT Levenberg-Marquart implementaton*****
#Define forward reflectance model here.
#Option to output model rrs and jacobian
#-----------------
def fitModel2(p, lam=None, fjac=None, rrsObs=None,aw=None, bbw=None, aphStar=None, \
              sdg=None,eta=None,g0=None,g1=None):

    if fjac is None:
        fjac_flag = 0;
    else:
        fjac_flag = 1;

    if len(p)==3:
        adgStar = ancIOP.get_adgStar(lam,sdg);
        bbpStar = ancIOP.get_bbpStar(lam,eta);
    elif len(p)==5:
        adgStar = ancIOP.get_adgStar(lam,p[3]);
        bbpStar = ancIOP.get_bbpStar(lam,p[4]);

    atot = aw + p[0]*aphStar + p[1]*adgStar;
    bbtot = bbw + p[2]*bbpStar;
    kappa = atot + bbtot;
    u = bbtot / kappa;
    rrsMod = (g0 + g1*u)*u;

    status=0;

    pderiv = jfunc2(p,lam,aw,bbw,aphStar,adgStar,bbpStar,g0,g1);

    if (fjac_flag):
        return([status, (rrsObs-rrsMod)],pderiv);
    else:
        return(status, (rrsObs-rrsMod));


#-----------------
def fitModel3(p, lam, rrsObs,aw, bbw, aphStar, sdg, eta, g0, g1):

    if len(p)==3:
        adgStar = ancIOP.get_adgStar(lam,sdg);
        bbpStar = ancIOP.get_bbpStar(lam,eta);
    elif len(p)==5:
        adgStar = ancIOP.get_adgStar(lam,p[3]);
        bbpStar = ancIOP.get_bbpStar(lam,p[4]);

    atot = aw + p[0]*aphStar + p[1]*adgStar;
    bbtot = bbw + p[2]*bbpStar;
    kappa = atot + bbtot;
    u = bbtot / kappa;
    rrsMod = (g0 + g1*u)*u;

    return  (rrsObs-rrsMod);


#-----------------
#Compute Jacobian (method 1 - for levmar and mpfit)
#-----------------
def jfunc2(p,lam,aw,bbw,aphStar,sdg,eta,g0,g1):

    if len(p)==3:
        adgStar = ancIOP.get_adgStar(lam,sdg);
        bbpStar = ancIOP.get_bbpStar(lam,eta);
    elif len(p)==5:
        adgStar = ancIOP.get_adgStar(lam,p[3]);
        bbpStar = ancIOP.get_bbpStar(lam,p[4]);

    atot = aw + p[0]*aphStar + p[1]*adgStar;
    bbtot = bbw + p[2]*bbpStar;
    kappa = atot + bbtot;
    u = bbtot / kappa;
    rrsMod = (g0 + g1*u)*u;

    pderiv = np.zeros([len(aw), len(p)]);

    jac0 = (g0 + 2.*g1*u)*(-bbtot*aphStar) / (atot+bbtot)**2.;
    jac1 = (g0 + 2.*g1*u)*(-bbtot*adgStar) / (atot+bbtot)**2;
    jac2 = (g0 + 2.*g1*u)*(atot*bbpStar)   / (atot+bbtot)**2.;
    if len(p)==5:
        i443 = np.where( np.fabs(lam - 443) == np.min(np.fabs(lam-443)))[0][0]
        jac3 = (g0 + 2.*g1*u)*(bbtot*p[1]*(lam-lam[i443])*adgStar) / (atot+bbtot)**2.;
        jac4 = (g0 + 2.*g1*u)*(atot*p[2]*np.log(lam[i443]/lam)*bbpStar) / (atot+bbtot)**2.;
    #
    pderiv[:,0] = jac0;
    pderiv[:,1] = jac1;
    pderiv[:,2] = jac2;
    if len(p)==5:
        pderiv[:,3] = jac3;
        pderiv[:,4] = jac4


    return pderiv;
#
#-----------------
#Cost function for leastsq optimization - note returns spectral difference value
#-----------------
def errFunc1(p0,lam,aw,bbw,aphStar,sdg,eta,g0,g1,rrsObs):
    rrsMod = fitModel(p0,lam,aw,bbw,aphStar,sdg,eta,g0,g1);
    err = np.copy(rrsObs - rrsMod);
    return err;
#
#-----------------
#Cost function for Amoeba optimization - note returns scalar sum of squares value
#-----------------
def errFunc2(p0,lam,aw,bbw,aphStar,sdg,eta,g0,g1,rrsObs,covRrs):
    rrsMod = fitModel(p0,lam,aw,bbw,aphStar,sdg,eta,g0,g1);
    err = np.copy(np.sum((rrsObs - rrsMod)**2.));
    return err;
#
#-----------------
#Cost function for Amoeba optimization with uncertainties - note returns scalar sum of squares (wrt to uncertainty) value
#-----------------
def errFunc3(p0,lam,aw,bbw,aphStar,sdg,eta,g0,g1,rrsObs,covRrs):
    rrsMod = fitModel(p0,lam,aw,bbw,aphStar,sdg,eta,g0,g1);
    err = np.copy(np.sum(((rrsObs - rrsMod)**2.)/np.diag(covRrs)));
    return err;
#
#-----------------
#Cost function for Amoeba optimization with uncertainties + Bayesian component (scalar output)
#-----------------
def errFuncBayes(p0,lam,aw,bbw,aphStar,sdg,eta,g0,g1,rrsObs,covRrs,prior,cov_prior):
    rrsMod = fitModel(p0,lam,aw,bbw,aphStar,sdg,eta,g0,g1);
    err_rrs = np.copy(np.sum(((rrsObs - rrsMod)**2.)/np.diag(covRrs)));
    err_pri = np.copy(np.sum(((p0-prior)**2.)/np.diag(cov_prior)));
    err = err_rrs + err_pri;
    return err;

#-----------------
#Levenberg-Marquart optimization (mpfit)
#-----------------
def lmSol(rrsObs,guess,lam,aw,bbw,aphStar,sdg,eta,g0,g1,badval=-999):
    #
    #Function keywords are input as a dictionary
    fa = {'rrsObs':rrsObs,'lam':lam,'aw':aw,'bbw':bbw,'aphStar':aphStar,'sdg':sdg,\
                                        'eta':eta,'g0':g0,'g1':g1};

    moutput = mpfit.mpfit(fitModel2, guess,autoderivative=0,functkw=fa,quiet=True);

    pOut = moutput.params;
    ier = moutput.status; #solution status (for mpfit, want it to +ve integer)

    if ier < 0:
        status = 1;
        pOut=[badval]*len(pOut)
    else:
        # Good values
        status = 0

    return status,pOut;
    ##
#

#------------------
#Linear matric inversion
#------------------
def lmiSol(rrsObs,lam,aw,bbw,aphStar,sdg,eta,g0,g1,badval=-999):
    adgStar = ancIOP.get_adgStar(lam,sdg);
    bbpStar = ancIOP.get_bbpStar(lam,eta);
    #
    #Postive quadratic root
    qRoot = (-g0 + np.sqrt(g0**2. + 4.* g1 * rrsObs)) / (2.* g1);
    #
    #set up matrix
    b1 = bbw * (1.0 - qRoot) - aw * qRoot;
    b1 = b1.T;
    A1 = np.array(((aphStar * qRoot), (adgStar * qRoot), (bbpStar * (qRoot - 1.0)) ));
    A1 = A1.T;
    #
    #QR decomposition
    Q, R = np.linalg.qr(A1);
    pOut =np.linalg.solve(R, np.linalg.solve(R.T,(np.dot(A1.T,b1))));
    r1 = b1 - np.dot(A1,pOut);
    err1 = np.linalg.solve(R, np.linalg.solve(R.T,np.dot(A1.T,r1)));
    pOut = pOut + err1;
    status = 0;
    #
    return status, pOut;

##
#------------------
#Nelder-Mead (Amoeba) optimization
#------------------
def amoebaSol(rrsObs,guess,lam,aw,bbw,aphStar,sdg,eta,g0,g1,covRrs,error='absolute',badval=-999):

    #Run optimization routine
    errfunc = errFunc2 if error=='absolute' else errFunc3

    output = minimize(errfunc, guess, args=(lam,aw,bbw,aphStar,sdg,eta,g0,g1,rrsObs,covRrs),\
                      method='Nelder-Mead',options={'fatol':1e-8,'xatol':1e-8,'maxfev':2000});
    pOut = output.x;

    #Is there a warning message?
    if output.status >0:
        status = 1;
        pOut = [badval]*len(guess)
    else:
        #Good values
        status = 0;
    return status,pOut;
##
#------------------
#Nelder-Mead (Amoeba) optimization with Bayesian wrapper
#------------------
def BayesSol(rrsObs,guess,lam,aw,bbw,aphStar,adgStar,bbpStar,g0,g1,covRrs,prior,cov_prior,badval=-999):

    #Run optimization routine
    output = minimize(errFuncBayes, guess, args=(lam,aw,bbw,aphStar,adgStar,bbpStar,g0,g1,rrsObs,covRrs,prior,cov_prior),\
                      method='Nelder-Mead',options={'fatol':1e-8,'xatol':1e-8,'maxfev':2000});
    pOut = output.x;

    #Is there a warning message?
    if output.status >0:
        status = 1;
        pOut = [badval]*len(guess)
    else:
        #Good values
        status = 0;
    return status,pOut;

#
#*******************************************************************************
