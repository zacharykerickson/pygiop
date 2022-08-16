#*******************************************************************************
#                           ----------------
#                               ancIOP.py
#                           ----------------
#
# Description: functions used to return spectra for non-phytoplankton
#   and non-pure water IOPs
#
# Zachary Erickson
# NASA OBPG, July 2021
#
#*******************************************************************************

import numpy as np;

def get_bbpStar(lam,eta):
    i443 = np.where( np.fabs(lam - 443) == np.min(np.fabs(lam-443)))[0][0]

    bbpStar = (lam[i443] / lam)**eta;

    return bbpStar;
##
def get_adgStar(lam,sdg):
    i443 = np.where( np.fabs(lam - 443) == np.min(np.fabs(lam-443)))[0][0]

    adgStar = np.exp(-sdg * (lam - lam[i443]));

    return adgStar;
##
