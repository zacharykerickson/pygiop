#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 13:08:55 2021

@author: lmckinna
"""

import numpy as np;
import matplotlib.pyplot as plt;
from . import inverseMethod as invert;
from . import ancIOP;

#-----------------------------
'''
#************Check c implementation in ocssw#***************
# Compute Moore-Penrose pseudo inverse of matrix
def mpinv(A,rcond=1E-15):

    m,n = A.shape;

    flip = 0;

    if (m > n):
        #Flip matrix
        flip = 1;
        A = A.T;
        i = m;
        m = n;
        n = i;
    ##

    #Compute SVD
    U, s, V = np.linalg.svd(A, full_matrices=True,hermitian=False);


    #Python vs C code
    # A gets overwritten as U
    # V is V
    # s is u

    thresh = np.max(s)*rcond;
    s[s<=thresh] = 0.0;

    sigma_pinv = np.zeros((n,m));

    for i in range(0,m):
        if s[i] > thresh:
            sigma_pinv[i,i] = 1./s[i];


    Unew = np.zeros((n,n));

    for i in range(0,m):
        for j in range(0,m):
            Unew[i,j] = U[i,j]


    temp = np.dot(sigma_pinv,U.T);
    Apinv = np.dot(V.T,temp);

    if (flip):
        Apinv = Apinv.T;

    return Apinv;
'''
#------------------------
# Compute uncertainties of freee parameters using Jacobian matrix
#
# covRrs: covariance matrix of sub-surface Remote sensing reflectances
# jac: jacobian matrixugeag
#-------------------------------
def calc_unc(covRrs,jac):

    pinvJ = np.linalg.pinv(jac,rcond=1e-5);

    covX = np.dot(pinvJ,np.dot(covRrs,pinvJ.T))

    uX = np.diagonal(covX)**0.5;

    return uX;

def calc_full_covar(covRrs,jac):

    pinvJ = np.linalg.pinv(jac,rcond=1e-5)

    convX = np.dot(pinvJ,np.dot(covRrs,pinvJ.T))

    return convX
#
#-----------------------------
# Compute uncertainties
#
# pOut: best fit model parameters
# aw: pure water absorption coefficient
# bbw: pure water backscattering coeffcient
# aphStar: phytoplankton normalised absorption coefficient
# adgStar: cdom + nap normalised absorption coefficient
# bbpStar:  normalised particle bascattering coefficient
# g0: gordon reflecntance model coefficeint g0
# g1: gordon reflectnce model coeffficient g1
# covRrs: covariance matrix of sub-surface Remote sensing reflectances
#-------------------------------
def get_giop_unc(pOut,lam,aw,bbw,aphStar,sdg,eta,g0,g1,covRrs):

    #Get the jacobian matrix
    jac = invert.jfunc2(pOut,lam,aw,bbw,aphStar,sdg,eta,g0,g1);

    if len(pOut)==3:
        adgStar = ancIOP.get_adgStar(lam,sdg);
        bbpStar = ancIOP.get_bbpStar(lam,eta);
    elif len(pOut)==5:
        adgStar = ancIOP.get_adgStar(lam,p[3]);
        bbpStar = ancIOP.get_bbpStar(lam,p[4]);

    #Estimate the paramterse uncertainties
    uX = calc_unc(covRrs,jac);

    uaph = ((aphStar*uX[0])**2.)**0.5; #simple for now - but it is more complex
    uadg = ((adgStar*uX[1])**2.)**0.5;
    ubbp = ((bbpStar*uX[2])**2.)**0.5; #need to add uncertainty for spectral slope

    uatot = (uaph**2 + uadg**2)**0.5;
    ubbtot = ubbp;

    return uatot,ubbtot,uX[0],uX[1],uX[2];
#
def get_giop_covar(pOut,lam,aw,bbw,aphStar,sdg,eta,g0,g1,covRrs):

    if len(pOut)==3:
        adgStar = ancIOP.get_adgStar(lam,sdg);
        bbpStar = ancIOP.get_bbpStar(lam,eta);
    elif len(pOut)==5:
        adgStar = ancIOP.get_adgStar(lam,pOut[3]);
        bbpStar = ancIOP.get_bbpStar(lam,pOut[4]);

    # Get the Jacobian matrix
    jac = invert.jfunc2(pOut,lam,aw,bbw,aphStar,sdg,eta,g0,g1);

    # Estimate the covariance matrix for the retrieved parameters
    covar = calc_full_covar(covRrs,jac);
    uatot = (aphStar**2*covar[0,0] + adgStar**2*covar[1,1] + 2*aphStar*adgStar*covar[0,1])**0.5;
    ubbtot = (bbpStar**2*covar[2,2])**0.5;

    return uatot,ubbtot,covar
#
def get_giop_covar_Bayes(pOut,lam,aw,bbw,aphStar,sdg,eta,g0,g1,covRrs,cov_prior):

    # Get the Jacobian matrix
    jac = invert.jfunc2(pOut,lam,aw,bbw,aphStar,sdg,eta,g0,g1);

    covar = np.linalg.inv(np.dot(jac.T,np.dot(np.linalg.inv(covRrs),jac)) + np.linalg.inv(cov_prior))

    return covar
#
