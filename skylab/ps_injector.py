# -*-coding:utf8-*-

from __future__ import print_function

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

ps_injector
===========

Point Source Injection classes. The interface with the core
PointSourceLikelihood - Class requires the methods

    fill - Filling the class with Monte Carlo events

    sample - get a weighted sample with mean number `mu`

    flux2mu - convert from a flux to mean number of expected events

    mu2flux - convert from a mean number of expected events to a flux

"""

# python packages
import logging

# scipy-project imports
import numpy as np
from numpy.lib.recfunctions import drop_fields
import scipy.interpolate

# local package imports
from . import set_pars
from .utils import rotate

############################################################################
## HealpyLLH extra imports
import healpy as hp
# My analysis tools
import anapymods.healpy as amp_hp
############################################################################

# get module logger
def trace(self, message, *args, **kwargs):
    r""" Add trace to logger with output level beyond debug

    """
    if self.isEnabledFor(5):
        self._log(5, message, args, **kwargs)

logging.addLevelName(5, "TRACE")
logging.Logger.trace = trace

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

_deg = 4
_ext = 3

def rotate_struct(ev, ra, dec):
    r"""Wrapper around the rotate-method in skylab.utils for structured
    arrays.

    Parameters
    ----------
    ev : structured array
        Event information with ra, sinDec, plus true information

    ra, dec : float
        Coordinates to rotate the true direction onto

    Returns
    --------
    ev : structured array
        Array with rotated value, true information is deleted

    """
    names = ev.dtype.names

    rot = np.copy(ev)

    # Function call
    rot["ra"], rot_dec = rotate(ev["trueRa"], ev["trueDec"],
                                ra * np.ones(len(ev)), dec * np.ones(len(ev)),
                                ev["ra"], np.arcsin(ev["sinDec"]))

    if "dec" in names:
        rot["dec"] = rot_dec
    rot["sinDec"] = np.sin(rot_dec)

    # "delete" Monte Carlo information from sampled events
    mc = ["trueRa", "trueDec", "trueE", "ow"]

    return drop_fields(rot, mc)


class Injector(object):
    r"""Base class for Signal Injectors defining the essential classes needed
    for the LLH evaluation.

    """

    def __init__(self, *args, **kwargs):
        r"""Constructor: Define general point source features here...

        """
        self.__raise__()

    def __raise__(self):
        raise NotImplementedError("Implemented as abstract in {0:s}...".format(
                                    self.__repr__()))

    def fill(self, *args, **kwargs):
        r"""Filling the injector with the sample to draw from, work only on
        data samples known by the LLH class here.

        """
        self.__raise__()

    def flux2mu(self, *args, **kwargs):
        r"""Internal conversion from fluxes to event numbers.

        """
        self.__raise__()

    def mu2flux(self, *args, **kwargs):
        r"""Internal conversion from mean number of expected neutrinos to
        point source flux.

        """
        self.__raise__()

    def sample(self, *args, **kwargs):
        r"""Generator method that returns sampled events. Best would be an
        infinite loop.

        """
        self.__raise__()


class PointSourceInjector(Injector):
    r"""Class to inject a point source into an event sample.

    """
    _src_dec = np.nan
    _sinDec_bandwidth = 0.1
    _sinDec_range = [-1., 1.]

    _E0 = 1.
    _GeV = 1.e3
    _e_range = [0., np.inf]

    _random = np.random.RandomState()
    _seed = None

    def __init__(self, gamma, **kwargs):
        r"""Constructor. Initialize the Injector class with basic
        characteristics regarding a point source.

        Parameters
        -----------
        gamma : float
            Spectral index, positive values for falling spectra

        kwargs : dict
            Set parameters of class different to default

        """

        # source properties
        self.gamma = gamma

        # Set all other attributes passed to the class
        set_pars(self, **kwargs)

        return

    def __str__(self):
        r"""String representation showing some more or less useful information
        regarding the Injector class.

        """
        sout = ("\n{0:s}\n"+
                67*"-"+"\n"+
                "\tSpectral index     : {1:6.2f}\n"+
                "\tSource declination : {2:5.1f} deg\n"
                "\tlog10 Energy range : {3:5.1f} to {4:5.1f}\n").format(
                         self.__repr__(),
                         self.gamma, np.degrees(self.src_dec),
                         *self.e_range)
        sout += 67*"-"

        return sout

    @property
    def sinDec_range(self):
        return self._sinDec_range

    @sinDec_range.setter
    def sinDec_range(self, val):
        if len(val) != 2:
            raise ValueError("SinDec range needs only upper and lower bound!")
        if val[0] < -1 or val[1] > 1:
            logger.warn("SinDec bounds out of [-1, 1], clip to that values")
            val[0] = max(val[0], -1)
            val[1] = min(val[1], 1)
        if np.diff(val) <= 0:
            raise ValueError("SinDec range has to be increasing")
        self._sinDec_range = [float(val[0]), float(val[1])]
        return

    @property
    def e_range(self):
        return self._e_range

    @e_range.setter
    def e_range(self, val):
        if len(val) != 2:
            raise ValueError("Energy range needs upper and lower bound!")
        if val[0] < 0. or val[1] < 0:
            logger.warn("Energy range has to be non-negative")
            val[0] = max(val[0], 0)
            val[1] = max(val[1], 0)
        if np.diff(val) <= 0:
            raise ValueError("Energy range has to be increasing")
        self._e_range = [float(val[0]), float(val[1])]
        return

    @property
    def GeV(self):
        return self._GeV

    @GeV.setter
    def GeV(self, value):
        self._GeV = float(value)

        return

    @property
    def E0(self):
        return self._E0

    @E0.setter
    def E0(self, value):
        self._E0 = float(value)

        return

    @property
    def random(self):
        return self._random

    @random.setter
    def random(self, value):
        self._random = value

        return

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, val):
        logger.info("Setting global seed to {0:d}".format(int(val)))
        self._seed = int(val)
        self.random = np.random.RandomState(self.seed)

        return

    @property
    def sinDec_bandwidth(self):
        return self._sinDec_bandwidth

    @sinDec_bandwidth.setter
    def sinDec_bandwidth(self, val):
        if val < 0. or val > 1:
            logger.warn("Sin Declination bandwidth {0:2e} not valid".format(
                            val))
            val = min(1., np.fabs(val))
        self._sinDec_bandwidth = float(val)

        self._setup()

        return

    @property
    def src_dec(self):
        return self._src_dec

    @src_dec.setter
    def src_dec(self, val):
        if not np.fabs(val) < np.pi / 2.:
            logger.warn("Source declination {0:2e} not in pi range".format(
                            val))
            return
        if not (np.sin(val) > self.sinDec_range[0]
                and np.sin(val) < self.sinDec_range[1]):
            logger.error("Injection declination not in sinDec_range!")
        self._src_dec = float(val)

        self._setup()

        return

    def _setup(self):
        r"""If one of *src_dec* or *dec_bandwidth* is changed or set, solid
        angles and declination bands have to be re-set.

        """

        A, B = self._sinDec_range

        m = (A - B + 2. * self.sinDec_bandwidth) / (A - B)
        b = self.sinDec_bandwidth * (A + B) / (B - A)

        sinDec = m * np.sin(self.src_dec) + b

        min_sinDec = max(A, sinDec - self.sinDec_bandwidth)
        max_sinDec = min(B, sinDec + self.sinDec_bandwidth)

        self._min_dec = np.arcsin(min_sinDec)
        self._max_dec = np.arcsin(max_sinDec)

        # solid angle of selected events
        self._omega = 2. * np.pi * (max_sinDec - min_sinDec)

        return

    def _weights(self):
        r"""Setup weights for given models.

        """
        # weights given in days, weighted to the point source flux
        self.mc_arr["ow"] *= self.mc_arr["trueE"]**(-self.gamma) / self._omega

        self._raw_flux = np.sum(self.mc_arr["ow"], dtype=np.float)

        # normalized weights for probability
        self._norm_w = self.mc_arr["ow"] / self._raw_flux

        # double-check if no weight is dominating the sample
        if self._norm_w.max() > 0.1:
            logger.warn("Warning: Maximal weight exceeds 10%: {0:7.2%}".format(
                            self._norm_w.max()))

        return

    def fill(self, src_dec, mc, livetime):
        r"""Fill the Injector with MonteCarlo events selecting events around
        the source position(s).

        Parameters
        -----------
        src_dec : float, array-like
            Source location(s)
        mc : recarray, dict of recarrays with sample enum as key (MultiPointSourceLLH)
            Monte Carlo events
        livetime : float, dict of floats
            Livetime per sample

        """

        if isinstance(mc, dict) ^ isinstance(livetime, dict):
            raise ValueError("mc and livetime not compatible")

        self.src_dec = src_dec

        self.mc = dict()
        self.mc_arr = np.empty(0, dtype=[("idx", np.int), ("enum", np.int),
                                         ("trueE", np.float), ("ow", np.float)])

        if not isinstance(mc, dict):
            mc = {-1: mc}
            livetime = {-1: livetime}

        for key, mc_i in mc.iteritems():
            # get MC event's in the selected energy and sinDec range
            band_mask = ((np.sin(mc_i["trueDec"]) > np.sin(self._min_dec))
                         &(np.sin(mc_i["trueDec"]) < np.sin(self._max_dec)))
            band_mask &= ((mc_i["trueE"] / self.GeV > self.e_range[0])
                          &(mc_i["trueE"] / self.GeV < self.e_range[1]))

            if not np.any(band_mask):
                print("Sample {0:d}: No events were selected!".format(key))
                self.mc[key] = mc_i[band_mask]

                continue

            self.mc[key] = mc_i[band_mask]

            N = np.count_nonzero(band_mask)
            mc_arr = np.empty(N, dtype=self.mc_arr.dtype)
            mc_arr["idx"] = np.arange(N)
            mc_arr["enum"] = key * np.ones(N)
            mc_arr["ow"] = self.mc[key]["ow"] * livetime[key] * 86400.
            mc_arr["trueE"] = self.mc[key]["trueE"]

            self.mc_arr = np.append(self.mc_arr, mc_arr)

            print("Sample {0:s}: Selected {1:6d} events at {2:7.2f}deg".format(
                        str(key), N, np.degrees(self.src_dec)))

        if len(self.mc_arr) < 1:
            raise ValueError("Select no events at all")

        print("Selected {0:d} events in total".format(len(self.mc_arr)))

        self._weights()

        return

    def flux2mu(self, flux):
        r"""Convert a flux to mean number of expected events.

        Converts a flux :math:`\Phi_0` to the mean number of expected
        events using the spectral index :math:`\gamma`, the
        specified energy unit `x GeV` and the point of normalization `E0`.

        The flux is calculated as follows:

        .. math::

            \frac{d\Phi}{dE}=\Phi_0\,E_0^{2-\gamma}
                                \left(\frac{E}{E_0}\right)^{-\gamma}

        In this way, the flux will be equivalent to a power law with
        index of -2 at the normalization energy `E0`.

        """

        gev_flux = (flux
                        * (self.E0 * self.GeV)**(self.gamma - 1.)
                        * (self.E0)**(self.gamma - 2.))

        return self._raw_flux * gev_flux

    def mu2flux(self, mu):
        r"""Calculate the corresponding flux in [*GeV*^(gamma - 1) s^-1 cm^-2]
        for a given number of mean source events.

        """

        gev_flux = mu / self._raw_flux

        return (gev_flux
                    * self.GeV**(1. - self.gamma) # turn from I3Unit to *GeV*
                    * self.E0**(2. - self.gamma)) # go from 1*GeV* to E0

    def sample(self, src_ra, mean_mu, poisson=True):
        r""" Generator to get sampled events for a Point Source location.

        Parameters
        -----------
        mean_mu : float
            Mean number of events to sample

        Returns
        --------
        num : int
            Number of events
        sam_ev : iterator
            sampled_events for each loop iteration, either as simple array or
            as dictionary for each sample

        Optional Parameters
        --------------------
        poisson : bool
            Use poisson fluctuations, otherwise sample exactly *mean_mu*

        """

        # generate event numbers using poissonian events
        while True:
            num = (self.random.poisson(mean_mu)
                        if poisson else int(np.around(mean_mu)))

            logger.debug(("Generated number of sources: {0:3d} "+
                          "of mean {1:5.1f} sources").format(num, mean_mu))

            # if no events should be sampled, return nothing
            if num < 1:
                yield num, None
                continue

            sam_idx = self.random.choice(self.mc_arr, size=num, p=self._norm_w)

            # get the events that were sampled
            enums = np.unique(sam_idx["enum"])

            if len(enums) == 1 and enums[0] < 0:
                # only one sample, just return recarray
                sam_ev = np.copy(self.mc[enums[0]][sam_idx["idx"]])

                yield num, rotate_struct(sam_ev, src_ra, self.src_dec)
                continue

            sam_ev = dict()
            for enum in enums:
                idx = sam_idx[sam_idx["enum"] == enum]["idx"]
                sam_ev_i = np.copy(self.mc[enum][idx])
                sam_ev[enum] = rotate_struct(sam_ev_i, src_ra, self.src_dec)

            yield num, sam_ev


class ModelInjector(PointSourceInjector):
    r"""PointSourceInjector that weights events according to a specific model
    flux.

    Fluxes are measured in percent of the input flux.

    """

    def __init__(self, logE, logFlux, *args, **kwargs):
        r"""Constructor, setting up the weighting function.

        Parameters
        -----------
        logE : array
            Flux Energy in units log(*self.GeV*)

        logFlux : array
            Flux in units log(*self.GeV* / cm^2 s), i.e. log(E^2 dPhi/dE)

        Other Parameters
        -----------------
        deg : int
            Degree of polynomial for flux parametrization

        args, kwargs
            Passed to PointSourceInjector

        """

        deg = kwargs.pop("deg", _deg)
        ext = kwargs.pop("ext", _ext)

        s = np.argsort(logE)
        logE = logE[s]
        logFlux = logFlux[s]
        diff = np.argwhere(np.diff(logE) > 0)
        logE = logE[diff]
        logFlux = logFlux[diff]

        self._spline = scipy.interpolate.InterpolatedUnivariateSpline(
                            logE, logFlux, k=deg)

        # use default energy range of the flux parametrization
        kwargs.setdefault("e_range", [10.**np.amin(logE), 10.**np.amax(logE)])

        # Set all other attributes passed to the class
        set_pars(self, **kwargs)

        return

    def _weights(self):
        r"""Calculate weights, according to given flux parametrization.

        """

        trueLogGeV = np.log10(self.mc_arr["trueE"]) - np.log10(self.GeV)

        logF = self._spline(trueLogGeV)
        flux = np.power(10., logF - 2. * trueLogGeV) / self.GeV

        # remove NaN's, etc.
        m = (flux > 0.) & np.isfinite(flux)
        self.mc_arr = self.mc_arr[m]

        # assign flux to OneWeight
        self.mc_arr["ow"] *= flux[m] / self._omega

        self._raw_flux = np.sum(self.mc_arr["ow"], dtype=np.float)

        self._norm_w = self.mc_arr["ow"] / self._raw_flux

        return

    def flux2mu(self, flux):
        r"""Convert a flux to number of expected events.

        """

        return self._raw_flux * flux

    def mu2flux(self, mu):
        r"""Convert a mean number of expected events to a flux.

        """

        return float(mu) / self._raw_flux


############################################################################
## HealpyLLH
############################################################################
class HealpyInjector(PointSourceInjector):
    r"""
    Injector for injecting events from healpy spatial pdf maps.

    Not using `src_dec` because source positions are given by the healpy map.
    For the same reason, `sinDec_bandwidth` is always the whole sky. Events
    get simply sampled according to the src_map, no need to restrcit that.

    Also `sinDec_bandwidth` has a slightly different purpose.

    Parameters
    ----------
    sinDec_bandwidth : float array
        sinDec_range[0] is the lower and sinDec_range[1] bound in sinDec in
        which MC events are select for injection.
        Make sure, that this span all the src position, otherwise
    """
    _src = None
    _sinDec_bandwidth = 0.1

    # Those are constant here becasue we sample from src maps directly
    _sinDec_range = [-1., +1.]
    _omega = 4. * np.pi

    _E0 = 1.
    _GeV = 1.e3
    _e_range = [0., np.inf]

    _random = np.random.RandomState()
    _seed = None


    def __init__(self, src, **kwargs):
        """
        Set src array as a attribute so we can use it for the `sample()`
        function.

        Also explicitely init 'sinDec_range' and 'sinDec_bandwidth' to set all
        dependent vars.

        Parameters
        ----------
        src is a numpy record array with the following fields needed:
            norm_w : array
                Total normed weight for each source. This includes theoretical
                and detector acceptance weight for each source. This field is
                calculated in the psLLH.HealpyLLH.use_sources()` function.
            sigma : array
               List of healpy maps. The maps are convolved with a gaussian with
               the reconstruction sigma of each event and used as the spatial
               pdf for each event. (dtype is object or numpy.ndarray)
        """
        self.src = src

        self.sinDec_bandwidth = kwargs.pop(
            "sinDec_bandwidth", self._sinDec_bandwidth)

        # Set solid angle and some other fixed stuff
        self._min_sinDec = self._sinDec_range[0]
        self._max_sinDec = self._sinDec_range[1]
        self._min_dec = np.arcsin(self._min_sinDec)
        self._max_dec = np.arcsin(self._max_sinDec)
        # Should always be 4pi
        self._omega = 2. * np.pi * (self._max_sinDec - self._min_sinDec)

        super(HealpyInjector, self).__init__(**kwargs)
        return

    def __str__(self):
        r"""
        String representation showing some more or less useful information
        regarding the Injector class.
        Removed `src_dec` as it is not needed here.
        """
        sout = ("\n{:s}\n"+
                67*"-"+"\n"+
                "\tSpectral index     : {:6.2f}\n"+
                "\tsinDec bandwidth   : {:5.1f}°\n"
                "\tlog10 Energy range : {:5.1f} to {:5.1f}\n").format(
                         self.__repr__(), self.gamma,
                         np.rad2deg(np.arcsin(self.sinDec_bandwidth)),
                         *self.e_range)
        sout += 67*"-"
        return sout


    # PROPERTIES
    # Overwrite some unused setters
    @property
    def sinDec_range(self):
        return self._sinDec_range
    @sinDec_range.setter
    def sinDec_range(self, val):
        print("sinDec_range is always [-1,+1] in HealpyInjector. Skipping.")
        return
    @property
    def src_dec(self):
        return self._src_dec
    @src_dec.setter
    def src_dec(self, val):
        print("src_dec is not used in HealpyInjector. Skipping.")
        return
    # Remove calling _setup() from sinDec_bandwidth setter. _omega stays 4pi
    @property
    def sinDec_bandwidth(self):
        return self._sinDec_bandwidth
    @sinDec_bandwidth.setter
    def sinDec_bandwidth(self, val):
        if val < 0. or val > 1.:
            print("sinDec_bandwidth must be in [0, 1]. Skipping")
            return
        self._sinDec_bandwidth = float(val)
        self._dec_bandwidth = np.arcsin(self._sinDec_bandwidth)
        return
    # The source recarray must be a class attribute to stay consistent with
    # the `fill()` function call
    @property
    def src(self):
        return self._src
    @src.setter
    def src(self, src):
       self._src = src
       return



    # PUBLIC METHODS
    def fill(self, src_dec, mc, livetime):
        r"""
        Fill the Injector with MonteCarlo events by sampling event positions
        from the spatial source map. Those positions are assigned to MC events
        in the same declination band as the sampled event corresponding to
        the events weight.

        Only slightly adapted to not use 'src_dec' and added 'ra', dec' and
        'idx' information to mc_arr, to use it in sample().

        If multiple samples are given the dictionary must have the structure
        `{key1:sample1, key2:sample2}` where keys are integers and samples are
        record arrays. The same applies to livetimes: `{key1:livetime1, ...}`.
        If only one sample is given as a record array it is internally
        converted to a dictionary with `{-1 : mc}` and `{-1 : livetime}`.

        `src_dec` is kept as a dummy argument for compatibility but has
        no effect.

        Parameters
        -----------
        mc : record array or dict of record arrays
            Monte Carlo event sample or multiple samples.
            Must be the same as the MC used in psLLH to cache the maps.
        livetime : float or dict of floats
            Livetime of the MC sample or mutliple samples.
        """
        if isinstance(mc, dict) ^ isinstance(livetime, dict):
            raise ValueError("mc and livetime not compatible")

        # enum * idx is a unique id for each event. In dicts the last key is
        # valid if two identical keys are given.
        self.mc = dict()
        # Add dec information for use in sample() function
        self.mc_arr = np.empty(0, dtype=[("idx", np.int), ("enum", np.int),
                                         ("trueE", np.float), ("ow", np.float),
                                         ("dec", np.float), ("ra", np.float)])

        if not isinstance(mc, dict):
            mc = {-1: mc}
            livetime = {-1: livetime}

        for key, mc_i in mc.iteritems():
            # get MC event's in the selected energy and sinDec range
            band_mask = ((np.sin(mc_i["trueDec"]) > np.sin(self._min_dec))
                         &(np.sin(mc_i["trueDec"]) < np.sin(self._max_dec)))
            band_mask &= ((mc_i["trueE"] / self.GeV > self.e_range[0])
                          &(mc_i["trueE"] / self.GeV < self.e_range[1]))

            if not np.any(band_mask):
                print("Sample {0:d}: No events were selected!".format(key))
                self.mc[key] = mc_i[band_mask]

                continue

            self.mc[key] = mc_i[band_mask]

            N = np.count_nonzero(band_mask)
            mc_arr = np.empty(N, dtype=self.mc_arr.dtype)
            mc_arr["idx"] = np.arange(N)
            mc_arr["enum"] = key * np.ones(N)
            mc_arr["ow"] = self.mc[key]["ow"] * livetime[key] * 86400.
            mc_arr["trueE"] = self.mc[key]["trueE"]
            # Add ra, dec information
            mc_arr["ra"] = self.mc[key]["ra"]
            mc_arr["dec"] = self.mc[key]["dec"]

            # Append 'idx' information to dict mc too to use it in sample()
            self.mc[key] = np.lib.recfunctions.append_fields(
                self.mc[key], "idx", mc_arr["idx"],
                dtypes=np.int, usemask=False)

            self.mc_arr = np.append(self.mc_arr, mc_arr)

            sout = (67 * "-" + "\nFill HealpyInjector info:"
                    "  Sample {:s}: Selected {:6d} events\n"
                    "    DEC : {:7.2f}° - {:7.2f}°\n"
                    "    E   : {:7.2f} and {:7.2f} in {:7.2f} GeV").format(
                        str(key), N,
                        np.rad2deg(self._min_dec), np.rad2deg(self._max_dec),
                        self.e_range[0], self.e_range[1], self.GeV)
            print(sout)

        if len(self.mc_arr) < 1:
            raise ValueError("Select no events at all")

        print("  Selected {0:d} events in total".format(len(self.mc_arr)))
        print(67*"-")

        self._weights()

        return


    # def sample(self, mean_mu, src_map, poisson=True):
    def sample(self, src_ra, mean_mu, poisson=True):
        r"""
        Inject mean_mu (more below) events using rejection sampling from
        the healpy src_map.

        Events from MC are selected from the same declination band as the
        sampled position because the detector acceptance is declination
        dependent and only events from the same declination band are
        representative for an event position at the same declination.

        `src_ra` is kept as a dummy argument for compatibility but has
        no effect.

        Parameters
        ----------
        mean_mu : float
            Mean number of events to sample
        poisson : bool
            Use poisson fluctuations, otherwise sample exactly *mean_mu*.
            (default: True)

        Returns
        --------
        num : int
            Number of events
        sam_ev : iterator
            Sampled events for each loop iteration, either as simple array or
            as dictionary for each sample, having the same fields as exp.
        """
        def assign_and_drop(sam_ev, inj_ra, inj_dec):
            r"""
            Assign sampled ra/dec positions and drop mc fields from
            injected sample.
            """
            # Assign sampled locations from src map. This replaces the rotate
            # function in PointSourceInjector
            sam_ev["ra"] = inj_ra
            sam_ev["dec"] = inj_dec
            # MC names to drop from the injected events
            mc_names = ['ow', 'trueDec', 'trueE', 'trueRa']
            return drop_fields(sam_ev, mc_names)

        # Here starts the generator part
        while True:
            # Sample mu_mean directly or get n_inj from a poisson distribution
            # if poisson = True was given
            n_inj = (self.random.poisson(mean_mu)
                if poisson else int(np.around(mean_mu)))
            # if no events should be sampled, return nothing
            if n_inj < 1:
                yield n_inj, None
                continue

            # First we need to select sources to sample from using their weight
            src_idx = np.random.choice(
                np.arange(len(self.src)), size=n_inj, p=src["norm_w"])

            # Sample indices from selected src maps and convert to map
            # coordinates
            inj_ind = np.zeros(n_inj)
            for i, src_id in enumerate(src_idx):
                # Sample one event from the selected source map
                inj_ind[i], _ = amp_hp.healpy_rejection_sampler(
                    src["sigma"][src_id], n=1)

            # Assign indices to healpy map coordinates
            inj_th, inj_phi = hp.pix2ang(NSIDE, inj_ind)

            # Get equatorial coordinates for the injected signal
            inj_dec, inj_ra = amp_hp.ThetaPhiToDecRa(inj_th, inj_phi)

            # Temporary recarray to hold the sampled mc_arr events
            sam_idx = np.empty((n_inj,), dtype=self.mc_arr.dtype)
            indices = np.arange(len(self.mc_arr))
            # Loop over injected evts, select MC evt from all given samples
            # and fill it into inj.
            for i, (_dec, ra) in enumerate(zip(inj_dec, inj_ra)):
                # First get the zenith band per event with correct boundaries
                # This is from psLLH _select_events()
                dec = (np.pi - 2. * self._dec_bandwidth) / np.pi * _dec
                min_dec = max(-np.pi / 2., dec - self._dec_bandwidth)
                max_dec = min(+np.pi / 2., dec + self._dec_bandwidth)

                # Pick a random event from the MC data in the dec band.
                # Events are weighted using the _norm_w set up in _weight()
                mask = np.logical_and(
                    self.mc_arr["dec"]>min_dec, self.mc_arr["dec"]<max_dec)
                _id = indices[mask]
                # Choose one event from mc_arr per sampled location
                sam_idx[i] = np.random.choice(
                    self.mc_arr, size=1, p=self._norm_w)

                # Sampled locations areeasily applied directly
                sam_idx["ra"][i] = inj_ra[i]
                sam_idx["dec"][i] = inj_dec[i]


            # get the events that were sampled from each sample seperately
            enums = np.unique(sam_idx["enum"])

            # Case if only rec array was given in fill()
            if len(enums) == 1 and enums[0] < 0:
                # only one sample, just return recarray
                sam_ev = np.copy(self.mc[enums[0]][sam_idx["idx"]])
                # Assign sampled ra/dec to single out array
                yield n_inj, assign_and_drop(sam_ev, inj_ra, inj_dec)
                continue

            # For multiple samples return each injected event in a dict
            # indicating its original sample
            sam_ev = dict()
            for enum in enums:
                mask = (sam_idx["enum"] == enum)
                idx = sam_idx[mask]["idx"]
                sam_ev_i = np.copy(self.mc[enum][idx])
                # Assign sampled ra/dec to each out array
                sam_ev[enum] = assign_and_drop(
                    sam_ev_i, sam_idx["ra"][mask], sam_idx["dec"][mask])

            yield n_inj, sam_ev

