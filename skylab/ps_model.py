# -*-coding:utf8-*-

from __future__ import print_function

'''
from icecube.photospline.glam import glam
#from icecube.photospline import spglam as glam
from icecube.photospline import splinefitstable
from icecube.photospline.glam.glam import grideval
from icecube.photospline.glam.bspline import bspline
'''
"""
This file is part of SkyLab

Skylab is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

# python packages
from copy import deepcopy
from itertools import product
import logging

# scipy-project imports
import numpy as np
import scipy.interpolate
from scipy.stats import norm
import scipy.sparse as sps

# local package imports
from . import set_pars
from .utils import kernel_func

# get module logger
def trace(self, message, *args, **kwargs):
    """ Add trace to logger with output level beyond debug
    """
    if self.isEnabledFor(5):
        self._log(5, message, args, **kwargs)

logging.addLevelName(5, "TRACE")
logging.Logger.trace = trace

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

# default values for parameters

# model-parameters
_gamma_params = dict(gamma=[2., (1., 4.)])

# histogramming
_sinDec_bins = 25
_sinDec_range = None
_2dim_bins = 25

# spline
_ratio_perc = 99.
_1dim_order = 2
_2dim_order = 2
_precision = 0.1
_par_val = np.nan
_parab_cache = np.zeros((0, ), dtype=[("S1", np.float), ("a", np.float),
                                      ("b", np.float)])


class NullModel(object):
    r"""Base class of models for likelihood fitting, this defines every core
    class of the likelihood fitting that is needed in the point source
    calculation without implementing any functionality. Use this class as
    starting point for a unbinned point source likelihood model

    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
                "NullModel only to be used as abstract superclass".format(
                    self.__repr__()))

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, val):
        self._params = val

        return

    @params.deleter
    def params(self):
        self._params = dict()

        return

    def __raise__(self, *args, **kwargs):
        raise NotImplementedError("Implemented as abstract in {0:s}...".format(
                self.__repr__()))

    def __call__(self, *args, **kwargs):
        r"""Calling the class-object will set it up for use of the other
        functions, i.e., creating splines out of data, etc.

        """
        self.__raise__()

    def background(self, *args, **kwargs):
        r"""Calculation of the background probability *B* in the point source
        likelihood, mainly a spatial dependent term.

        """
        self.__raise__()

    def signal(self, *args, **kwargs):
        r"""Calculation of the signal probability *S* in the point source
        likelihood, mainly a spatial dependent term.

        """
        self.__raise__()

    def reset(self, *args, **kwargs):
        r"""Resetting the llh-model to delete possible cached values

        """

        self.__raise__()

    def weight(self, *args, **kwargs):
        r"""Additional weights calculated for each event, commonly used to
        implement energy weights in the point source likelihood.

        It differs from signal and background distributions that is (to
        first approximation) does not depend on the source position.

        """
        self.__raise__()


class ClassicLLH(NullModel):
    r"""Classic likelihood model for point source searches, only using spatial
    information of each event

    """

    sinDec_bins = _sinDec_bins
    sinDec_range = _sinDec_range

    _order = _1dim_order

    _bckg_spline = np.nan

    _gamma = 2.

    def __init__(self, *args, **kwargs):
        r"""Constructor of ClassicLLH. Set all configurations here.

        """

        self.params = kwargs.pop("params", dict())

        # Set all attributes passed to class
        set_pars(self, **kwargs)

        return

    def __call__(self, exp, mc, livetime, **kwargs):
        r"""Use experimental data to create one dimensional spline of
        declination information for background information.

        Parameters
        -----------
        exp : structured array
            Experimental data with all neccessary fields, i.e., sinDec for
            ClassicLLH
        mc : structured array
            Same as exp for Monte Carlo plus true information.
        livetime : float
            Livetime to scale the Monte Carlo with

        """

        hist, bins = np.histogram(exp["sinDec"], density=True,
                                  bins=self.sinDec_bins,
                                  range=self.sinDec_range)

        # background spline

        # overwrite range and bins to actual bin edges
        self.sinDec_bins = bins
        self.sinDec_range = (bins[0], bins[-1])

        if np.any(hist <= 0.):
            bmids = (self.sinDec_bins[1:] + self.sinDec_bins[:-1]) / 2.
            estr = ("Declination hist bins empty, this must not happen! "
                    +"Empty bins: {0}".format(bmids[hist <= 0.]))
            raise ValueError(estr)
        elif np.any((exp["sinDec"] < bins[0])|(exp["sinDec"] > bins[-1])):
            raise ValueError("Data outside of declination bins!")

        self._bckg_spline = scipy.interpolate.InterpolatedUnivariateSpline(
                                (bins[1:] + bins[:-1]) / 2.,
                                np.log(hist), k=self.order)

        # eff. Area
        self._effA(mc, livetime, **kwargs)

        return

    def __str__(self):
        r"""String representation of ClassicLLH.

        """
        out_str = "{0:s}\n".format(self.__repr__())
        out_str += 67*"~"+"\n"
        out_str += "Spatial background hist:\n"
        out_str += "\tSinDec bins  : {0:3d}\n".format(len(self.sinDec_bins)-1)
        out_str += "\tSinDec range : {0:-4.2f} to {1:-4.2f}\n".format(
                        *self.sinDec_range)
        out_str += 67*"~"+"\n"

        return out_str

    def _effA(self, mc, livetime, **kwargs):
        r"""Build splines for effective Area given a fixed spectral
        index *gamma*.

        """

        # powerlaw weights
        w = mc["ow"] * mc["trueE"]**(-self.gamma) * livetime * 86400.

        # get pdf of event distribution
        h, bins = np.histogram(np.sin(mc["trueDec"]), weights=w,
                               bins=self.sinDec_bins, density=True)

        # normalize by solid angle
        h /= np.diff(self.sinDec_bins)

        # multiply histogram by event sum for event densitiy
        h *= w.sum()

        self._spl_effA = scipy.interpolate.InterpolatedUnivariateSpline(
                (bins[1:] + bins[:-1]) / 2., np.log(h), k=self.order)

        return

    @property
    def bckg_spline(self):
        return self._bckg_spline

    @bckg_spline.setter
    def bckg_spline(self, val):
        if not hasattr(val, "__call__"):
            print(">>> WARNING: {0} is not callable! Not spline-ish".format(val))
            return

        self._bckg_spline = val

        return

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, val):
        self._gamma = float(val)

        return

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, val):
        self.order = int(val)

        return

    def background(self, ev):
        r"""Spatial background distribution.

        For IceCube is only declination dependent, in a more general scenario,
        it is dependent on zenith and
        azimuth, e.g. in ANTARES, KM3NET, or using time dependent information.

        Parameters
        -----------
        ev : structured array
            Event array, importand information *sinDec* for this calculation

        Returns
        --------
        P : array-like
            spatial background probability for each event to be found
            at *sinDec*

        """
        return 1. / 2. / np.pi * np.exp(self.bckg_spline(ev["sinDec"]))

    def effA(self, dec, **params):
        r"""Calculate integrated effective Area at declination for distributing
        source events among different samples.

        """

        # if (np.sin(dec) < self.sinDec_bins[0]
        #         or np.sin(dec) > self.sinDec_bins[-1]):
        #     return 0., None

        sinDec = np.sin(np.atleast_1d(dec))

        valid = ((sinDec >= self.sinDec_bins[0]) &
                 (sinDec <= self.sinDec_bins[-1]))
        effA = np.zeros_like(sinDec)
        effA = self._spl_effA(sinDec[valid])

        return effA, None

    def reset(self):
        r"""Classic likelihood does only depend on spatial part, needs no
        caching

        """
        return
    #@profile
    def fast_signal(self, src_ra, src_dec, ev, ind):
        r"""Spatial distance between source position and events

        Signal is assumed to cluster around source position.
        The distribution is assumed to be well approximated by a gaussian
        locally.

        Parameters
        -----------
        ev : structured array
            Event array, import information: sinDec, ra, sigma

        Returns
        --------
        P : array-like
            Spatial signal probability for each event

        """

        cos_ev = np.sqrt(1. - ev["sinDec"]**2)
        cosDist = (np.cos(src_ra - np.take(ev['ra'],ind))
                * np.cos(src_dec) * np.take(cos_ev,ind)
                + np.sin(src_dec) * np.take(ev["sinDec"],ind))


        # handle possible floating precision errors
        cosDist[np.isclose(cosDist, 1.) & (cosDist > 1)] = 1.
        dist = np.arccos(cosDist)

        sigma = np.take(ev['sigma'],ind)
        result = (1./2./np.pi/sigma**2
                * np.exp(-dist**2 / 2. / sigma**2))

        return result


    def signal(self, src_ra, src_dec, ev):
        r"""Spatial distance between source position and events

        Signal is assumed to cluster around source position.
        The distribution is assumed to be well approximated by a gaussian
        locally.

        Parameters
        -----------
        ev : structured array
            Event array, import information: sinDec, ra, sigma

        Returns
        --------
        P : array-like
            Spatial signal probability for each event

        """



        #convert src_ra, dec to numpy arrays if not already done
        src_ra = np.atleast_1d(src_ra)[:,np.newaxis]
        src_dec = np.atleast_1d(src_dec)[:,np.newaxis]

        cos_ev = np.sqrt(1. - ev["sinDec"]**2)
        cosDist = (np.cos(src_ra - ev["ra"])
                * np.cos(src_dec) * cos_ev
                + np.sin(src_dec) * ev["sinDec"])


        # handle possible floating precision errors
        cosDist[np.isclose(cosDist, 1.) & (cosDist > 1)] = 1.
        dist = np.arccos(cosDist)

        return (1./2./np.pi/ev["sigma"]**2
                * np.exp(-dist**2 / 2. / ev["sigma"]**2))


    def weight(self, ev, **params):
        r"""For classicLLH, no weighting of events

        """
        return np.ones(len(ev)), None


class UniformLLH(ClassicLLH):
    r"""Spatial LLH class that assumes uniform distribution.

    """

    def __call__(self, *args, **kwargs):
        return

    def background(self, ev):
        return np.full(len(ev), 1. / 4. / np.pi)


class WeightLLH(ClassicLLH):
    r"""Likelihood class supporting weights for the calculation.

    The weights are calculated using N observables for exp. data and Monte
    Carlo.

    Abstract class, not incorporating a weighting scheme for Monte Carlo.

    """

    _precision = _precision

    _g1 = _par_val
    _w_cache = _parab_cache

    def __init__(self, params, pars, bins, *args, **kwargs):
        r"""Constructor

        Parameters
        -----------
        params : dict
            List of fit parameters. Each entry is a tuple out of
            (seed, [lower bound, upper bound])
        pars : list
            Parameter names to use for histogram, without sinDec, which is
            added as last normalisation parameter
        bins : int, ndarray
            Binning for each parameter

        Other Parameters
        -----------------
        range : ndarray
            Bin ranges for each parameter
        kernel : ndarray, int, float
            Smoothing filter defining the kernel for smoothing. Smoothing done
            solely for dimensions that are not normalised. A ndarray specifies
            the filter directly, an int is used for a flat kernel with size
            of *filter* in direction of both sides, a float uses a normal
            distributed kernel with approximately one standard deviation per
            bin.

        """

        params = params

        self.hist_pars = pars
        #print(self.hist_pars)

        self._ndim_bins = bins
        self._ndim_range = kwargs.pop("range", None)
        self._ndim_norm = kwargs.pop("normed", 0)

        # define kernel
        kernel = kwargs.pop("kernel", 0)
        if np.all(np.asarray(kernel) == 0):
            # No smoothing
            self._XX = None
        elif isinstance(kernel, (list, np.ndarray)):
            kernel_arr = np.asarray(kernel)
            assert(np.all(kernel_arr >= 0))
            XX = np.meshgrid(*([kernel_arr for i in range(len(self.hist_pars)
                                                          - self._ndim_norm)]
                              + [[1] for i in range(self._ndim_norm)]))
            self._XX = np.product(XX, axis=0).T
        elif isinstance(kernel, int):
            assert(kernel > 0)
            kernel_arr = np.ones(2 * kernel + 1, dtype=np.float)
            XX = np.meshgrid(*([kernel_arr for i in range(len(self.hist_pars)
                                                          - self._ndim_norm)]
                              + [[1] for i in range(self._ndim_norm)]))
            self._XX = np.product(XX, axis=0).T
        elif isinstance(kernel, float):
            assert(kernel >= 1)
            val = 1.6635
            r = np.linspace(-val, val, 2 * int(kernel) + 1)
            kernel_arr = norm.pdf(r)
            XX = np.meshgrid(*([kernel_arr for i in range(len(self.hist_pars)
                                                          - self._ndim_norm)]
                              + [[1] for i in range(self._ndim_norm)]))
            self._XX = np.product(XX, axis=0).T
        else:
            raise ValueError("Kernel has to be positive int / float or array")

        super(WeightLLH, self).__init__(*args, params=params, **kwargs)

        self._w_spline_dict = dict()

        return

    def __call__(self, exp, mc, livetime):
        r"""In addition to *classicLLH.__call__(),
        splines for energy-declination are created as well.

        """

        self._setup(exp)
        #self._multi_spline(mc)

        # calclate splines for all values of splines
        par_grid = dict()
        for par, val in self.params.iteritems():
            # create grid of all values that could come up due to boundaries
            # use one more grid point below and above for gradient calculation
            low, high = val[1]
            grid = np.arange(low - self._precision,
                             high + 2. * self._precision,
                             self._precision)
            par_grid[par] = grid

        pars = par_grid.keys()
        for tup in product(*par_grid.values()):
            # call spline function to cache the spline
            self._ratio_spline(mc, **dict([(p_i, self._around(t_i))
                                           for p_i, t_i in zip(pars, tup)]))

        # create spatial splines of classic LLH class and eff. Area
        super(WeightLLH, self).__call__(exp, mc, livetime, **par_grid)

        return

    def __str__(self):
        r"""String representation

        """
        out_str = super(WeightLLH, self).__str__()
        out_str += "Weighting hist:\n"
        for p, b, r in zip(self.hist_pars, self._ndim_bins, self._ndim_range):
            out_str += "\t{0:11s} : {1:3d}\n".format(p + " bins", len(b)-1)
            out_str += "\t{0:11s} : {1:-4.2f} to {2:-4.2f}\n".format(
                                        p + " range", *r)
        out_str += "\tPrecision : {0:4.2f}\n".format(self._precision)
        out_str += 67*"~"+"\n"

        return out_str

    def _around(self, value):
        r"""Round a value to a precision defined in the class.

        Parameters
        -----------
        value : array-like
            Values to round to precision.

        Returns
        --------
        round : array-like
            Rounded values.

        """
        return np.around(float(value) / self._precision) * self._precision

    def _get_weights(self, **params):
        r"""Calculate weights using the given parameters.

        Parameters
        -----------
        params : dict
            Dictionary containing the parameter values for the weighting.

        Returns
        --------
        weights : array-like
            Weights for each event

        """

        raise NotImplementedError("Weigthing not specified, using subclass")

    def _hist(self, arr, weights=None):
        r"""Create histogram of data so that it is correctly normalized.

        The edges of the histogram are copied so that the spline is defined for
        the entire data range.

        """

        h, binedges = np.histogramdd(arr, bins=self._ndim_bins,
                                     range=self._ndim_range,
                                     weights=weights, normed=True)

        if self._ndim_norm > 0:
            norms = np.sum(h, axis=tuple(range(h.ndim - self._ndim_norm)))
            norms[norms==0] = 1.

            h /= norms

        return h, binedges

    def _multi_spline(self, mc):
        r"""Create the ratio of signal over background probabilities. With same
        binning, the bin hypervolume cancels out, ensuring correct
        normalisation of the histograms.

        Parameters
        -----------
        mc : recarray
            Monte Carlo events to use for spline creation
        params : dict
            (Physics) parameters used for signal pdf calculation.

        Returns
        --------
        spline : scipy.interpolate.RectBivariateSpline
            Spline for parameter values *params*

        """

        mcvars = [mc[p] if not p == "sinDec" else np.sin(mc["trueDec"])
                  for p in self.hist_pars]

        par_grid = dict()
        for par, val in self.params.iteritems():
            # create grid of all values that could come up due to boundaries
            # use one more grid point below and above for gradient calculation
            low, high = val[1]
            grid = np.arange(low - self._precision,
                             high + 2. * self._precision,
                             self._precision)
            par_grid[par] = grid

        #print(par_grid)
        shape = [len(bins)-1 for bins in self._ndim_bins]
        #print(shape,[len(val) for val in par_grid.itervalues()])
        shape = tuple(shape+[len(val) for val in par_grid.itervalues()])
        #print(shape)
        #create empty d-dim matrix
        ratio = np.empty(shape,dtype=np.float)
        w = np.empty_like(ratio,dtype=np.float)

        for i,(par,val) in enumerate(par_grid.iteritems()):
            for j,gamma in enumerate(val):
                params = dict(gamma=gamma)

                # create MC histogram
                wSh, wSb = self._hist(mcvars, weights=self._get_weights(mc, **params))
                wSh = kernel_func(wSh, self._XX)
                wSd = wSh > 0.

                # calculate ratio
                r = np.ones_like(self._wB_hist, dtype=np.float)
                w_i = np.ones_like(r, dtype=np.float)

                r[wSd & self._wB_domain] = (wSh[wSd & self._wB_domain]
                                        / self._wB_hist[wSd & self._wB_domain])

                # values outside of the exp domain, but inside the MC one are mapped to
                # the most signal-like value
                min_ratio = np.percentile(r[r>1.], _ratio_perc)
                print('min:', min_ratio)
                np.copyto(r, min_ratio, where=wSd & ~self._wB_domain)
                np.copyto(w_i,0.1,where=wSd & ~self._wB_domain)

                ratio[:,:,j] = r
                w[:,:,j] = w_i


                if (gamma > 1.95) and (gamma < 2.05):
                    import matplotlib.pylab as plt
                    from matplotlib.colors import LogNorm
                    fig,ax = plt.subplots()
                    #print(r)
                    X, Y = np.meshgrid(wSb[0][1:], wSb[1][1:],indexing='ij')
                    #print(X.shape,Y.shape,r.shape)
                    import matplotlib.colors as colors
                    class MidpointNormalize(colors.LogNorm):
                        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
                            self.midpoint = midpoint
                            colors.LogNorm.__init__(self, vmin, vmax, clip)

                        def __call__(self, value, clip=None):
                                                    # I'm ignoring masked values and all kinds of edge cases to make a
                                                            # simple example...
                            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
                            return np.ma.masked_array(np.interp(value, x, y))

                    im=ax.pcolormesh(Y, X, r,cmap=plt.cm.RdBu_r,norm=MidpointNormalize(vmin=0.1,vmax=1.e2,midpoint=1.))
                    #ax.set_aspect('equal')
                    #print('here:',[r_i for r_i in r],r.shape)
                    ax.set_ylim([1,9])
                    ax.set_xlim([-1,1])
                    fig.colorbar(im)
                    plt.savefig('/Users/mhuber/test_hist.pdf')
                    plt.close()
                    break
        '''
        binmids = [(wSb_i[1:] + wSb_i[:-1]) / 2. for wSb_i in wSb]
        #print('mids:',binmids,np.array(binmids).shape)

        binmids[-1][[0, -1]] = wSb_i[0], wSb_i[-1]
        binmids = tuple(binmids+[grid for grid in par_grid.itervalues()])
        #print(binmids,binmids)

        order = 3

        knots = [np.linspace(bins[0]-0.1*np.abs(np.diff(bins[[0,-1]])),bins[-1]+0.1*np.abs(np.diff(bins[[0,-1]])),20) for bins in binmids]
        smooth = 3.14159e3
        w = np.ones_like(ratio)
        w[ratio == 1.] = 0.25




        self._photospline = glam.fit(np.log(ratio),w,binmids,knots,order,smooth,penalties={2:[0,0,1],3:[1,1,0]})

        '''
        binmids = [(wSb_i[1:] + wSb_i[:-1]) / 2. for wSb_i in wSb]
        binmids[-1][[0, -1]] = wSb_i[0], wSb_i[-1]
        #binmids = tuple(binmids+[grid for grid in par_grid.itervalues()])
        #print(binmids)
        binmids= tuple(binmids)
        order = 3
        knots = [np.linspace(bins[0]-0.1*np.abs(np.diff(bins[[0,-1]])),bins[-1]+0.1*np.abs(np.diff(bins[[0,-1]])),35) for bins in binmids]
        #knots = [np.linspace(bins[0]-0.5,bins[-1]+0.5,35) for bins in binmids]
        smooth = 3.14159e1
        w_i[r == 1.] = 0.1

        w = np.ones_like(r)

        self._photospline = glam.fit(np.log(r),w_i,binmids,knots,order,smooth,penalties={3:[1,1]})



        return






    def _ratio_spline(self, mc, **params):
        r"""Create the ratio of signal over background probabilities. With same
        binning, the bin hypervolume cancels out, ensuring correct
        normalisation of the histograms.

        Parameters
        -----------
        mc : recarray
            Monte Carlo events to use for spline creation
        params : dict
            (Physics) parameters used for signal pdf calculation.

        Returns
        --------
        spline : scipy.interpolate.RectBivariateSpline
            Spline for parameter values *params*

        """

        mcvars = [mc[p] if not p == "sinDec" else np.sin(mc["trueDec"])
                  for p in self.hist_pars]


        # create MC histogram
        wSh, wSb = self._hist(mcvars, weights=self._get_weights(mc, **params))
        wSh = kernel_func(wSh, self._XX)
        wSd = wSh > 0.

        # calculate ratio
        ratio = np.ones_like(self._wB_hist, dtype=np.float)

        ratio[wSd & self._wB_domain] = (wSh[wSd & self._wB_domain]
                                        / self._wB_hist[wSd & self._wB_domain])

        # values outside of the exp domain, but inside the MC one are mapped to
        # the most signal-like value
        min_ratio = np.percentile(ratio[ratio>1.], _ratio_perc)
        np.copyto(ratio, min_ratio, where=wSd & ~self._wB_domain)

        binmids = [(wSb_i[1:] + wSb_i[:-1]) / 2. for wSb_i in wSb]
        binmids[-1][[0, -1]] = wSb_i[0], wSb_i[-1]
        binmids = tuple(binmids)

        '''
        if (params['gamma'] > 1.95) and (params['gamma'] < 2.05):
                    import matplotlib.pylab as plt
                    fig,ax = plt.subplots()

                    X, Y = np.meshgrid(wSb[0][1:], wSb[1][1:],indexing='ij')
                    #print(X.shape,Y.shape,r.shape)
                    ax.pcolormesh(Y, X, ratio,cmap=plt.cm.RdBu_r)
                    #ax.set_aspect('equal')
                    #print('here:',[r_i for r_i in r],r.shape)
                    #ax.colorbar()
                    plt.savefig('/Users/mhuber/test_hist2.pdf')
                    plt.close()
        '''

        spline = scipy.interpolate.RegularGridInterpolator(
                                                binmids, np.log(ratio),
                                                method="linear",
                                                bounds_error=False,
                                                fill_value=0.)

        self._w_spline_dict[tuple(params.items())] = spline

        return spline


    def _setup(self, exp):
        r"""Set up everything for weight calculation.

        """
        # set up weights for background distribution, reset all cached values
        self._w_spline_dict = dict()

        expvars = [exp[p] for p in self.hist_pars]

        self._wB_hist, self._wB_bins = self._hist(expvars)
        self._wB_hist = kernel_func(self._wB_hist, self._XX)
        self._wB_domain = self._wB_hist > 0

        # overwrite bins
        self._ndim_bins = self._wB_bins
        self._ndim_range = tuple([(wB_i[0], wB_i[-1])
                                  for wB_i in self._wB_bins])

        return

    def _spline_eval(self, spline, ev):
        r"""Evaluate spline on coordinates using the important parameters.

        """
        return spline(np.vstack([ev[p] for p in self.hist_pars]).T)

    @property
    def hist_pars(self):
        return self._hist_pars

    @hist_pars.setter
    def hist_pars(self, val):
        self._hist_pars = list(val)

        return

    def reset(self):
        r"""Energy weights are cached, reset all cached values.

        """
        self._w_cache = _parab_cache

        return


    def weight_new(self, ev, **params):
        r"""Evaluate spline for given parameters.

        Parameters
        -----------
        ev : structured array
            Events to be evaluated

        params : dict
            Parameters for evaluation

        Returns
        --------
        val : array-like (N), N events
            Function value.

        grad : array-like (N, M), N events in M parameter dimensions
            Gradients at function value.

        """
        # get params
        gamma = params["gamma"]

        coords = np.array([ev[p] if not p == "trueDec" else ev["sinDec"]
                  for p in self.hist_pars]).T

        #print(coords)
        #print([np.append(coord, gamma)[:,np.newaxis] for coord in coords][0:100])
        #print(glam.grideval(self._photospline, [[2],[0.5],[2]]))
        #val = [glam.grideval(self._photospline, np.append(coord, gamma)[:,np.newaxis]).ravel()[0] for coord in coords]
        val = [glam.grideval(self._photospline, coord[:,np.newaxis]).ravel()[0] for coord in coords]



        return np.exp(val)


    def weight(self, ev, **params):
        r"""Evaluate spline for given parameters.

        Parameters
        -----------
        ev : structured array
            Events to be evaluated

        params : dict
            Parameters for evaluation

        Returns
        --------
        val : array-like (N), N events
            Function value.

        grad : array-like (N, M), N events in M parameter dimensions
            Gradients at function value.

        """
        # get params
        gamma = params["gamma"]

        # evaluate on finite gridpoints in spectral index gamma
        g1 = self._around(gamma)
        dg = self._precision

        # check whether the grid point of evaluation has changed
        if (np.isfinite(self._g1)
                and g1 == self._g1
                and len(ev) == len(self._w_cache)):
            S1 = self._w_cache["S1"]
            a = self._w_cache["a"]
            b = self._w_cache["b"]
        else:
            # evaluate neighbouring gridpoints and parametrize a parabola
            g0 = self._around(g1 - dg)
            g2 = self._around(g1 + dg)

            S0 = self._spline_eval(self._w_spline_dict[(("gamma", g0), )], ev)
            S1 = self._spline_eval(self._w_spline_dict[(("gamma", g1), )], ev)
            S2 = self._spline_eval(self._w_spline_dict[(("gamma", g2), )], ev)

            a = (S0 - 2. * S1 + S2) / (2. * dg**2)
            b = (S2 - S0) / (2. * dg)

            # cache values
            self._g1 = g1

            self._w_cache = np.zeros((len(ev),),
                                     dtype=[("S1", np.float), ("a", np.float),
                                            ("b", np.float)])
            self._w_cache["S1"] = S1
            self._w_cache["a"] = a
            self._w_cache["b"] = b

        # calculate value at the parabola
        val = np.exp(a * (gamma - g1)**2 + b * (gamma - g1) + S1)
        grad = val * (2. * a * (gamma - g1) + b)

        return val, np.atleast_2d(grad)


class PowerLawLLH(WeightLLH):
    r"""Weighted LLH class assuming unbroken power-law spectra for weighting.

    Optional Parameters
    --------------------
    seed : float
        Seed for gamma parameter
    bonds : ndarray (len 2)
        Bounds for minimisation

    """
    def __init__(self, *args, **kwargs):

        params = dict(gamma=(kwargs.pop("seed", _gamma_params["gamma"][0]),
                             deepcopy(kwargs.pop("bounds", deepcopy(_gamma_params["gamma"][1])))))

        super(PowerLawLLH, self).__init__(params, *args, **kwargs)

        return

    def _effA(self, mc, livetime, **pars):
        r"""Calculate two dimensional spline of effective Area versus
        declination and spectral index for Monte Carlo.

        """

        gamma_vals = pars["gamma"]

        x = np.sin(mc["trueDec"])
        hist = np.vstack([np.histogram(x,
                                       weights=self._get_weights(mc, gamma=gm)
                                                * livetime * 86400.,
                                       bins=self.sinDec_bins)[0]
                          for gm in gamma_vals]).T

        # normalize bins by their binvolume, one dimension is the parameter
        # with width of *precision*
        bin_vol = np.diff(self.sinDec_bins)
        hist /= bin_vol[:, np.newaxis] * np.full_like(gamma_vals, self._precision)

        self._spl_effA = scipy.interpolate.RectBivariateSpline(
                (self.sinDec_bins[1:] + self.sinDec_bins[:-1]), gamma_vals,
                np.log(hist), kx=2, ky=2, s=0)

        return

    @staticmethod
    def _get_weights(mc, **params):
        r"""Calculate weights using the given parameters.

        Parameters
        -----------
        params : dict
            Dictionary containing the parameter values for the weighting.

        Returns
        --------
        weights : array-like
            Weights for each event

        """

        return mc["ow"] * mc["trueE"]**(-params["gamma"])

    def effA(self, dec, **params):
        r"""Evaluate effective Area at declination and spectral index.

        Parameters
        -----------
        dec : float
            Declination.

        gamma : float
            Spectral index.

        Returns
        --------
        effA : float
            Effective area at given point(s).
        grad_effA : float
            Gradient at given point(s).

        """
        dec = np.atleast_1d(dec)
        mask = (np.sin(dec) < self.sinDec_bins[0])|(np.sin(dec) > self.sinDec_bins[-1])
        #if np.any(mask):
        #    logger.warn('{0:3d} of {1:3d} sources outside of data declination range!'.format(np.count_nonzero(mask),len(mask)))

        if (np.all(mask) and (len(dec) == 1)):
            return 0., None

        gamma = params["gamma"]

        val = np.exp(self._spl_effA(np.sin(dec), gamma, grid=False, dy=0.))
        grad = val * self._spl_effA(np.sin(dec), gamma, grid=False, dy=1.)

        #set the effA and gradient of all sources outside the bin range to 0
        val[mask] = 0.
        grad[mask] = 0.

        return val, dict(gamma=grad)


class EnergyLLH(PowerLawLLH):
    r"""Likelihood using Energy Proxy and declination, where declination is
    used for normalisation to account for changing energy distributions.

    """
    def __init__(self, twodim_bins=_2dim_bins, twodim_range=None,
                 **kwargs):
        r"""Constructor

        """
        super(EnergyLLH, self).__init__(["logE", "sinDec"],
                                        twodim_bins, range=twodim_range,
                                        normed=1,
                                        **kwargs)

        return


class EnergyDistLLH(PowerLawLLH):
    r"""Likelihood using Energy Proxy and starting distance for evaluation.
    Declination is not used for normalisation assuming that the energy does not
    change rapidly with declination.

    """
    def __init__(self, twodim_bins=_2dim_bins, twodim_range=None,
                 **kwargs):
        r"""Constructor

        """
        super(EnergyDistLLH, self).__init__(["logE", "dist"],
                                            twodim_bins, range=twodim_range,
                                            **kwargs)

        return


class EnergyLLHfixed(EnergyLLH):
    r"""Energy Likelihood that uses external data to create the splines, and
    splines are not evaluated using the data given by call method.

    """
    def __init__(self, exp, mc, livetime, **kwargs):
        r"""Constructor

        """

        # call constructor of super-class, settings are set.
        super(EnergyLLHfixed, self).__init__(**kwargs)

        # do the call already
        super(EnergyLLHfixed, self).__call__(exp, mc, livetime)

        return

    def __call__(self, exp, mc, livetime):
        r"""Call function not used here

        """

        print("EnergyLLH with FIXED splines used here, call has no effect")

        return


