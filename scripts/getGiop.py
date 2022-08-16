#*******************************************************************************
#                           ---------------
#                               giop.py
#                           ---------------
#
# Description - Generalized Inherent Optical Properties (GIOP) ocean color
#               inversion algorithm
#
# Reference:  P.J. Werdell and 18 co-authors, "Generalized ocean color
#               inversion model for retrieving marine inherent optical
#               properties," Appl. Opt. 52, 2019-2037 (2013).
#
# Translated from P.J Werdell's Matlab code to Python by:
# Lachlan McKinna, NASA GSFC
# July, 2014
#
# Edited by Zachary Erickson, NOAA PMEL
# 2021, to add in a Bayesian implementation
# August, 2022 to clean up
#
#*******************************************************************************
# Import python modules
#Import distributed python modules
import os, sys,csv,gc;
import numpy as np;

#Import written modules
from . import pwIOP;
from . import phytoA;
from . import inverseMethod as invert;
from . import giopUncertainties as giopUnc;
from . import ancIOP;
#--------------------
#-----
def giop(lam,Rrs,covRrs,
         fq='gordon',eta='qaa',sdg='qaa',aph='bricaud',error='absolute',chl=0.2,
         inv='amoeba',output_jac=False,badval=-999,num_retr=3,print_first_guess=False,
         prior=[0,0,0],cov_prior=np.zeros(shape=(3,3)),
         trans=None,Sf=0.5,qc=0.3
         ):
    #
    #lam: wavelength array
    #Rrs: above water remote sensing reflectance array
    #covRrs: error-Covaraince matrix for Rrs
    #
    #----
    # Check some optional arguments
    if cov_prior.ndim==1:
        cov_prior = np.diag(cov_prior)

    if ~np.isin(num_retr,[3,5]):
        print ('-E- only 3 or 5 retrieved parameters are supported');
        sys.exit()
    ##
    if fq=='gordon':
        g0 = 0.0949;
        g1 = 0.0794;
    else:
        g0,g1 = fq;

    #----

    #--------------------------
    #Wavelength indices
    i412 = np.argmin(np.abs(lam - 412.));
    i443 = np.argmin(np.abs(lam - 443.));
    i555 = np.argmin(np.abs(lam - 555.));
    for i,wv in zip([i412,i443,i555],[412,443,555]):
        if np.abs(lam[i]-wv)>10:
            print('-E- Closest wavelength to %d is over 10 nm away (%d)'%(wv,lam[i]))
            sys.exit()
    #
    #--------------------------
    #take Rrs across the air-sea interface (default to Lee)
    #print 'calcualting sub-sruface rrs'
    rin = Rrs / 0.529 if trans=='flat' else Rrs / (0.52 + 1.7 * Rrs);

    #--------------------------
    #Get pure water coefficients
    #print 'getting pure water IOPs'
    aw = pwIOP.get_aw(lam);
    bbw = pwIOP.get_bbw(lam);
    #---------------------------
    #define eta (default to QAA)
    if isinstance(eta,str):
        if eta=='qaa':
            #from QAA version 5 (Lee et al. 2002, etc.)
            eta = 2.0 * (1.0 - 1.2 * np.exp(-0.9 * rin[i443]/rin[i555]));
        elif eta=='gsm':
            #% from GSM (Maritorena et al. 2002)
            eta = 1.03373
        else:
            print('-E- Do not understand "eta" as %s'%eta)
            sys.exit()
    ##
    #-----------------------------
    #Define Sdg (default to 0.018)
    if isinstance(sdg,str):
        if sdg=='qaa':
            #from QAA version 5 (Lee et al. 2002, etc.)
            sdg = 0.015 + 0.002 / (0.6 + rin[i443]/rin[i555]);
        elif sdg=='obpg':
            #homegrown product from NASA OBPG using NOMAD v2 (unpublished)
            sdg = 0.015 + 0.0038 * np.log10(Rrs[i412]/Rrs[i555]);
            if sdg <= 0.01:
                sdg = 0.01;
                ##
            if sdg >= 0.02:
                sdg = 0.02;
                ##
        elif sdg=='gsm':
            #from GSM (Maritorena et al. 2002)
            sdg = 0.02061;
        else:
            print('-E- Do not understand "sdg" as %s'%sdg)
            sys.exit(0)
    elif sdg is None:
        sdg = 0.018

    #
    if aph == 'bricaud':
        #Define aph (default to Bricaud 1998)
        #Bricaud 1998 spectra with aph*(443) normalized to 0.055 m2/mg
        bap,baps,baph,baphs = phytoA.get_bricaud_aph(chl,lam,1);
        aphStar = baphs;
        #
    elif aph == 'gsm':
        #from GSM (Maritorena et al. 2002)
        aphStar =  phytoA.get_gsm_aph(lam)
        #
    elif aph == 'ciotti':
        # from Ciotti and Bricaud 2006
        aphStar = phytoA.get_ciotti_aph(lam,SfMix=Sf);
    else:
        #User defined array
        aphStar = kwargs['aph'];
    ##

    aph_initial = chl # this seems to work with real MODISA data at least...
    # This factor is from looking at the difference between normalized and non-normalized aphStar from the bricaud phytoA spectrum
    # And seems to work reasonably well
    # But who knows...probably not actually correct

    if num_retr==3:
        p0 = np.array([aph_initial,0.01,0.001]); # initial guess
    elif num_retr==5:
        p0 = np.array([aph_initial,0.01,0.001,sdg,eta])

    if print_first_guess:
        print(p0)

    if inv == 'lm':
        #Use levenberg-marquart least squares optimization solution method
        status,pOut = invert.lmSol(rin,p0,lam,aw,bbw,aphStar,sdg,eta,g0,g1,badval=badval);

    elif inv == 'lmi':
        #use linear matrix inversion solution method
        status,pOut = invert.lmiSol(rin,lam,aw,bbw,aphStar,sdg,eta,g0,g1,badval=badval);

    elif inv == 'amoeba':
        #use amoeba (Nelder-Mead) optimization
        status,pOut = invert.amoebaSol(rin,p0,lam,aw,bbw,aphStar,sdg,eta,g0,g1,covRrs,error=error,badval=badval);

    elif inv == 'Bayes':
        #use Bayesian optimization (uses amoeba (Nelder-Mead))
        status,pOut = invert.BayesSol(rin,p0,lam,aw,bbw,aphStar,sdg,eta,g0,g1,covRrs,prior,cov_prior,badval=badval);

    else:
        print ('-E- %s inversion type not supported'%inv);
        sys.exit()

    ##

    #-----------------------------
    #QA/QC of outputs

    #Initialize outputs as badval
    mRrs_out = np.ones((len(lam),))*badval;
    atot_out = np.ones((2,len(lam)))*badval;
    bbtot_out = np.ones((2,len(lam)))*badval;
    adg_out = np.ones((2,))*badval;
    aph_out = np.ones((2,))*badval;
    bbp_out = np.ones((2,))*badval;
    sdg_out = badval;
    eta_out = badval;
    covar = np.ones((len(pOut),len(pOut)))*badval

    #If actual retrieval, reconstruct spectra
    if (status == 0):
        #
        rrsRecon = invert.fitModel(pOut,lam,aw,bbw,aphStar,sdg,eta,g0,g1);
        idxRange = np.where((lam > 400) & (lam < 600))[0];
        rtest =  np.mean(np.fabs(rrsRecon[idxRange] - rin[idxRange]) / rin[idxRange]);
        #
        #Does the forward-modelled Rrs agree with input Rrs to within 30 %?
        if rtest <= qc:

            aw_temp = aw;
            bbw_temp = bbw;
            sdg_out = sdg;
            eta_out = eta;
            mRrs_out = (0.52*rrsRecon) / (1.0 - 1.7*rrsRecon);

            maph = pOut[0];
            madg = pOut[1];
            mbbp = pOut[2];

            ##Check to see if spectral IOPs fall within the acceptable range;
            idxBad1 = np.where((maph < -0.005) | (maph > 5.0))[0];
            idxBad2 = np.where((madg < -0.005) | (madg > 5.0))[0];
            idxBad3 = np.where((mbbp < -0.005) | (mbbp > 5.0))[0];
            #
            if idxBad1.any():
                maph = badval;
                aw_temp[idxBad1] = 0;
                bbw_temp[idxBad1] = 0;
                uaph = badval;
                uatot[idxBad1] = badval;
                #
            if idxBad2.any():
                madg = badval;
                aw_temp[idxBad2] = 0;
                bbw_temp[idxBad2] = 0;
                uadg = badval;
                uatot[idxBad1] = badval;
                #
            if idxBad3.any():
                mbbp = badval;
                aw_temp[idxBad3] = 0;
                bbw_temp[idxBad3] = 0;
                ubbp = badval;
                ubbtot[idxBad1] = badval;
            ##

            adgStar = ancIOP.get_adgStar(lam,sdg)
            bbpStar = ancIOP.get_bbpStar(lam,eta)

            #Output spectral IOP values
            aph_out[0] = maph;
            adg_out[0] = madg;
            bbp_out[0] = mbbp;
            atot_out[0]= maph*aphStar + madg*adgStar + aw_temp;
            bbtot_out[0] = bbw_temp + mbbp*bbpStar;

            #Compute IOP covariances
            uatot,ubbtot,covar = giopUnc.get_giop_covar(pOut,lam,aw,bbw,aphStar,sdg,eta,g0,g1,covRrs);

            #Output spectral IOP uncertainty values
            atot_out[1] = uatot;
            bbtot_out[1] = ubbtot;


    ##
    if inv=='Bayes':
        covar_Bayes = giopUnc.get_giop_covar_Bayes(pOut,lam,aw,bbw,aphStar,sdg_out,eta_out,g0,g1,covRrs,cov_prior)
        if output_jac==True:
            jac = invert.jfunc2(pOut,lam,aw,bbw,aphStar,sdg_out,eta_out,g0,g1);
            return pOut,covar,atot_out,bbtot_out,sdg_out,eta_out,mRrs_out,covar_Bayes,jac
        return pOut,covar,atot_out,bbtot_out,sdg_out,eta_out,mRrs_out,covar_Bayes
    else:
        if output_jac==True:
            jac = invert.jfunc2(pOut,lam,aw,bbw,aphStar,sdg_out,eta_out,g0,g1);
            return pOut,covar,atot_out,bbtot_out,sdg_out,eta_out,mRrs_out,jac
        return pOut,covar,atot_out,bbtot_out,sdg_out,eta_out,mRrs_out

    #
##
#------------------------
#------------------------

#*******************************************************************************
