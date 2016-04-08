"""
Various codes to work with the initial mass function. Stolen shamelessly from 
Adam Ginsburg's agpy code:
https://code.google.com/p/agpy/source/browse/trunk/agpy/imf.py
"""
from __future__ import print_function, division, absolute_import
import types # I use typechecking.  Is there a better way to do this?  (see inverse_imf below)

import numpy as np


class MassFunction(object):
    """
    Generic Mass Function class
    """

    def dndm(self, m, **kwargs):
        """
        The differential form of the mass function, d N(M) / dM
        """
        return self(m, integral_form=False, **kwargs)

    def n_of_m(self, m, **kwargs):
        """
        The integral form of the mass function, N(M)
        """
        return self(m, integral_form=True, **kwargs)

    def integrate(self, mlow, mhigh, **kwargs):
        """
        Integrate the mass function over some range
        """
        import scipy.integrate
        return scipy.integrate.quad(self, mlow, mhigh, **kwargs)

class Salpeter(MassFunction):

    def __init__(self, alpha=2.35):
        """
        Create a default Salpeter mass function, i.e. a power-law mass function 
        the Salpeter 1955 IMF: dn/dm ~ m^-2.35
        """
        self.alpha = alpha

    def __call__(self, m, integral_form=False):
        if integral_form:
            return m**(-(self.alpha - 1))
        else:
            return m**(-self.alpha)


# three codes for dn/dlog(m)
salpeter = Salpeter()

class BrokenPowerLaw(MassFunction):
    def __init__(self, breaks, mmin, mmax):
        self.breaks = breaks
        self.normalization = self.integrate(mmin, mmax)[0]

    def __call__(self, m, integral_form=False):
        zeta = 0
        b_low = 0
        alp_low = 0
        for ii,b in enumerate(self.breaks):
            if integral_form:
                alp = self.breaks[b] - 1
            else:
                alp = self.breaks[b]
            if b == 'last':
                zeta += m**(-alp) * (b_low**(-alp+alp_low)) * (m>b_low)
            else:
                mask = ((m<b)*(m>b_low))
                zeta += m**(-alp) * (b**(-alp+alp_low)) *mask
                alp_low = alp
                b_low = b

        if hasattr(self,'normalization'):
            return zeta/self.normalization
        else:
            return zeta
            
#kroupa = BrokenPowerLaw(breaks={0.08:-0.3, 0.5:1.3, 'last':2.3},mmin=0.03,mmax=120)

class Kroupa(MassFunction):
    def __init__(self, mmin=0.03):
        """
        """
        self.mmin = mmin

    def __call__(self, m, p1=0.3, p2=1.3, p3=2.3, break1=0.08, break2=0.5, integral_form=False):
        """
        Kroupa 2001 IMF (http://arxiv.org/abs/astro-ph/0009005, http://adsabs.harvard.edu/abs/2001MNRAS.322..231K)
        """

        m = np.array(m)

        binv = ((break1**(-(p1-1)) - self.mmin**(-(p1-1)))/(1-p1) +
                (break2**(-(p2-1)) - break1**(-(p2-1))) * (break1**(p2-p1))/(1-p2) +
                (- break2**(-(p3-1))) * (break1**(p2-p1)) * (break2**(p3-p2))/(1-p3))
        b = 1./binv
        c = b * break1**(p2-p1)
        d = c * break2**(p3-p2)

        zeta = (b*(m**(-(p1))) * (m<break1) +
                c*(m**(-(p2))) * (m>=break1) * (m<break2) +
                d*(m**(-(p3))) * (m>=break2))

        if integral_form:
            return zeta * m
        else:
            return zeta

kroupa = Kroupa()


def chabrier(m, integral=False):
    """
    Chabrier 2003 IMF
    http://adsabs.harvard.edu/abs/2003PASP..115..763C
    (only valid for m < 1 msun)

    not sure which of these to use...

    integral is NOT IMPLEMENTED
    """
    if integral: print("Chabrier integral NOT IMPLEMENTED")
    # This system MF can be parameterized by the same type of lognormal form as
    # the single MF (eq. [17]), with the same normalization at 1 Msun, with the
    # coefficients (Chabrier 2003)
    return 0.86 * np.exp(-1*(np.log10(m)-np.log10(0.22))**2/(2*0.57**2))
    # This analytic form for the disk MF for single objects below 1 Msun, within these uncertainties, is given by the following lognormal form (Chabrier 2003):
    return 0.158 * np.exp(-1*(np.log10(m)-np.log10(0.08))**2/(2*0.69**2))

def schechter(m,A=1,beta=2,m0=100, integral=False):
    """
    A Schechter function with arbitrary defaults
    (integral may not be correct - exponent hasn't been dealt with at all)

    $$ A m^{-\\beta} e^{-m/m_0} $$
    
    Parameters
    ----------
        m : np.ndarray
            List of masses for which to compute the Schechter function
        A : float
            Arbitrary amplitude of the Schechter function
        beta : float
            Power law exponent
        m0 : float
            Characteristic mass (mass at which exponential decay takes over)
    
    Returns
    -------
        p(m) - the (unnormalized) probability of an object of a given mass
        as a function of that object's mass
        (though you could interpret mass as anything, it's just a number)

    """
    if integral: beta -= 1
    return A*m**-beta * np.exp(-m/m0)

def modified_schechter(m, m1, **kwargs):
    """
    A Schechter function with a low-level exponential cutoff
    "
    Parameters
    ----------
        m : np.ndarray
            List of masses for which to compute the Schechter function
        m1 : float
            Characteristic minimum mass (exponential decay below this mass)
        ** See schecter for other parameters ** 

    Returns
    -------
        p(m) - the (unnormalized) probability of an object of a given mass
        as a function of that object's mass
        (though you could interpret mass as anything, it's just a number)
    """
    return schechter(m, **kwargs) * np.exp(-m1/m)

try: 
    import scipy
    def schechter_cdf(m,A=1,beta=2,m0=100,mmin=10,mmax=None,npts=1e4):
        """
        Return the CDF value of a given mass for a set mmin,mmax
        mmax will default to 10 m0 if not specified
        
        Analytic integral of the Schechter function:
        http://www.wolframalpha.com/input/?i=integral%28x^-a+exp%28-x%2Fm%29+dx%29
        """
        if mmax is None:
            mmax = 10*m0
        
        # integrate the CDF from the minimum to maximum 
        # undefined posint = -m0 * mmax**-beta * (mmax/m0)**beta * scipy.special.gammainc(1-beta, mmax/m0)
        # undefined negint = -m0 * mmin**-beta * (mmin/m0)**beta * scipy.special.gammainc(1-beta, mmin/m0)
        posint = -mmax**(1-beta) * scipy.special.expn(beta, mmax/m0)
        negint = -mmin**(1-beta) * scipy.special.expn(beta, mmin/m0)
        tot = posint-negint

        # normalize by the integral
        # undefined ret = (-m0 * m**-beta * (m/m0)**beta * scipy.special.gammainc(1-beta, m/m0)) / tot
        ret = (-m**(1-beta) * scipy.special.expn(beta, m/m0) - negint)/ tot

        return ret

    def sh_cdf_func(**kwargs):
        return lambda x: schechter_cdf(x, **kwargs)
except ImportError:
    pass




#def schechter_inv(m): 
#    """
#    Return p(m)
#    """
#    return scipy.interpolate.interp1d(shfun,arange(.1,20,.01),bounds_error=False,fill_value=20.)

def integrate(fn=kroupa, bins=np.logspace(-2,2,500)):
    xax = (bins[:-1]+bins[1:])/2.
    integral = (bins[1:]-bins[:-1]) * (fn(bins[:-1])+fn(bins[1:])) / 2.

    return xax,integral

def m_integrate(fn=kroupa, bins=np.logspace(-2,2,500)):
    xax = (bins[:-1]+bins[1:])/2.
    integral = xax*(bins[1:]-bins[:-1]) * (fn(bins[:-1])+fn(bins[1:])) / 2.

    return xax,integral

def cumint(fn=kroupa, bins=np.logspace(-2,2,500)):
    xax,integral = integrate(fn,bins)
    return integral.cumsum() / integral.sum()

def m_cumint(fn=kroupa, bins=np.logspace(-2,2,500)):
    xax,integral = m_integrate(fn,bins)
    return integral.cumsum() / integral.sum()

massfunctions = {'kroupa':kroupa, 'salpeter':salpeter, 'chabrier':chabrier, 'schechter':schechter,'modified_schechter':modified_schechter}
if hasattr(massfunctions, '__iteritems__'):
    reverse_mf_dict = {v:k for k,v in massfunctions.iteritems()}
else:
    reverse_mf_dict = {v:k for k,v in massfunctions.items()}
# salpeter and schechter selections are arbitrary
mostcommonmass = {'kroupa':0.08, 'salpeter':0.01, 'chabrier':0.23, 'schecter':0.01,'modified_schechter':0.01}

def get_massfunc(massfunc):
    if type(massfunc) is types.FunctionType or hasattr(massfunc,'__call__'):
        return massfunc
    elif type(massfunc) is str:
        return massfunctions[massfunc]
    else:
        raise ValueError("massfunc must either be a string in the set %s or a function" % (",".join(massfunctions.keys())))

def get_massfunc_name(massfunc):
    if massfunc in reverse_mf_dict:
        return reverse_mf_dict[massfunc]
    elif type(massfunc) is str:
        return massfunc
    elif hasattr(massfunc,'__name__'):
        return massfunc.__name__
    else:
        raise ValueError("invalid mass function")

def inverse_imf(p, nbins=1000, mmin=0.03, mmax=120, massfunc='kroupa', **kwargs):
    """
    Inverse mass function

    massfunc can be 'kroupa', 'chabrier', 'salpeter', 'schechter', or a function
    """
 
    masses = np.logspace(np.log10(mmin),np.log10(mmax),nbins)
    mf = get_massfunc(massfunc)(masses, integral_form=True, **kwargs)
    mfcum = mf.cumsum()
    mfcum /= mfcum.max() # normalize to sum (cdf)

    return np.interp(p, mfcum, masses)