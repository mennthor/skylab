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
# StackingPointSourceLLH
import healpy as hp
import anapymods.healpy as amp_hp  # My analysis tools
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
# StackingPointSourceLLH
############################################################################
class StackingPointSourceInjector(PointSourceInjector):
    _src_dec = np.nan
    _sinDec_bandwidth = 0.1
    _sinDec_range = [-1., 1.]

    _E0 = 1.
    _GeV = 1.e3
    _e_range = [0., np.inf]

    _random = np.random.RandomState()
    _seed = None

    _src = None
    _src_priors = None
    _src_norm_w = None
    _nsrcs = 0

    _sinDec_bins = 25
    _spl_effA = None
    _order = 2

    mc = None
    mc_sel = None

    # Private Functions
    def _src_dec_weight_spline(self):
        """
        Same as in ps_model, make interpolating spline from sinDec hist.
        Independent of concrete event selection and needs only be recalculated
        for new MCs or if gamma is changed.

        Same function as in ps_model.
        """
        # Concat all needed information from all given MC samples. MCs should
        # describe all the same physics, so they can be treated as one single
        # MC sample. If this isn't the case, this function must be changed and
        # for every sample a different set of src detector weight must be
        # caluclated and used in the sample method.
        ow = []
        trueE = []
        trueDec = []
        livetime = 0.
        for key in self.mc.iterkeys():
            ow.append(self.mc[key]["ow"])
            trueE.append(self.mc[key]["trueE"])
            trueDec.append(self.mc[key]["trueDec"])
            livetime += self.livetime[key]
        ow = np.array(ow)
        trueE = np.array(trueE)
        trueDec = np.array(trueDec)


        # Powerlaw weights from NuGen simulation's OneWeight. The livetime is
        # not needed, because only one sample is used.
        w = mc["ow"] * mc["trueE"]**(-self.gamma)

        # Get event distribution dependent on declination. This is already
        # properly normalized to area by the `density` keyword
        h, bins = np.histogram(np.sin(mc["trueDec"]), weights=w,
            range=self._sinDec_range, bins=self.sinDec_bins, density=True)

        # Make interpolating spline through bin mids of histogram
        mids = 0.5 * (bins[1:] + bins[:-1])
        self._spl_src_dec_weights = \
            scipy.interpolate.InterpolatedUnivariateSpline(
                mids, h, k=self.order)


        # Powerlaw weights. Livetime given in days, converted to seconds
        w = ow * trueE**(-self.gamma) * livetime * 86400.
        # Get pdf of event distribution
        h, self.sinDec_bins = np.histogram(np.sin(trueDec), weights=w,
                                           range=self._sinDec_range,
                                           bins=self._sinDec_bins,
                                           density=True)
        # Normalize by solid angle
        h /= np.diff(self.sinDec_bins)
        # Multiply histogram by event sum for event densitiy
        h *= np.sum(w)
        # Interpolating spline. Attention: spline is in log(h)
        binmids = 0.5 * (self.sinDec_bins[:-1] + self.sinDec_bins[1:])

        # NaN handling
        y = np.log(h)
        w = np.isfinite(y)
        y = y[w]
        binmids = binmids[w]

        self._spl_effA = scipy.interpolate.InterpolatedUnivariateSpline(
            binmids, y, k=self._order)
        return

    def _setup(self, src_dec):
        """Vectorized to work for multiple src decs"""
        A, B = self._sinDec_range

        m = (A - B + 2. * self.sinDec_bandwidth) / (A - B)
        b = self.sinDec_bandwidth * (A + B) / (B - A)

        sinDec = m * np.sin(src_dec) + b

        min_sinDec = np.maximum(A, sinDec - self.sinDec_bandwidth)
        max_sinDec = np.minimum(B, sinDec + self.sinDec_bandwidth)

        self._min_dec = np.atleast_1d(np.arcsin(min_sinDec))
        self._max_dec = np.atleast_1d(np.arcsin(max_sinDec))

        # Solid angles of selected events
        self._omega = np.atleast_1d(2. * np.pi * (max_sinDec - min_sinDec))
        return

    def _select_events(self, src_dec):
        """Select events from the original MC dict for the current src dec
        positions that are sampled in sample() and store in a new array."""
        if self.mc is None:
            raise ValueError("Need to fill() MC events first.")

        # src_enum saves for which source the event was selected
        self.mc_arr = np.empty(0, dtype=[("idx", np.int),
                                         ("enum", np.int),
                                         ("src_enum", np.int),
                                         ("dec", np.float),
                                         ("trueE", np.float),
                                         ("ow", np.float)])

        # Init class dict to store selected events per source. First only copy
        # primary MC sample structure.
        # Structure is: 1. MC Sample, 2. Source Index, 3. MC array per source
        self.mc_sel = {}
        for key in self.mc.iterkeys():
            self.mc_sel[key] = {}

        # For every source j select events in source's declination band,
        # calculated in `_setup()`, from every MC sample in given dict.
        print(67 * "-" + "\nStackingLLHInjector fill() info:")
        for j, (omega, min_dec, max_dec) in enumerate(zip(
                self._omega, self._min_dec, self._max_dec)):
            print("  Source {:2d}".format(j))
            print("    DEC : {:7.2f}° - {:7.2f}°".format(
                np.rad2deg(min_dec), np.rad2deg(max_dec)))
            print("    E   : {:7.2f} and {:7.2f} in {:7.2f} GeV".format(
                self.e_range[0], self.e_range[1], self.GeV))
            # Now iterate over all MC samples and select events for the src j
            for key, mc_i in self.mc.iteritems():
                # Get MC event's in the selected energy and sinDec range for
                # the current source
                band_mask = ((np.sin(mc_i["trueDec"]) > np.sin(min_dec)) &
                             (np.sin(mc_i["trueDec"]) < np.sin(max_dec)))
                band_mask &= ((mc_i["trueE"] / self.GeV > self.e_range[0]) &
                              (mc_i["trueE"] / self.GeV < self.e_range[1]))

                if not np.any(band_mask):
                    print("    Sample {0:d}: No events ".format(key) +
                          "were selected for source {:2d}".format(j))
                    # 'Save' zero events but skip appending to array
                    self.mc_sel[key][j] = mc_i[band_mask]
                    continue

                # Safe selected events per source in class dict
                self.mc_sel[key][j] = mc_i[band_mask]

                # Append all selected events to a single record array
                N = np.count_nonzero(band_mask)
                mc_arr = np.empty(N, dtype=self.mc_arr.dtype)
                # idx, enum, src_enum identifies each event uniquely
                mc_arr["idx"] = np.arange(N)
                mc_arr["enum"] = key * np.ones(N)
                mc_arr["src_enum"] = j * np.ones(N)
                # Scale each OW to its corresponding livetime for the sample
                mc_arr["ow"] = (self.mc_sel[key][j]["ow"] *
                                self.livetime[key] * 86400.)
                mc_arr["trueE"] = self.mc_sel[key][j]["trueE"]
                mc_arr["dec"] = self.mc_sel[key][j]["dec"]

                self.mc_arr = np.append(self.mc_arr, mc_arr)

                print("    Sample {0:s}: Selected {1:6d} events".format(
                    str(key), N))

        if len(self.mc_arr) < 1:
            raise ValueError("Select no events at all")

        print("Selected {0:d} events in total".format(len(self.mc_arr)))

        # Update event and src weights for current selection
        self._weights(src_dec)

        return

    def _weights(self, src_dec):
        """Version for multiple srcs. Calculate src total normed weight and
        event weights with the injected raw flux for given srcs."""
        # First calculate total src weights = det. weight * theo. src weight
        src_norm_w = self.effA(src_dec) * self._src["src_w"]
        self._src_norm_w = src_norm_w / np.sum(src_norm_w)

        # Init for loop below
        self._raw_flux_per_src = np.zeros(len(self._src), dtype=np.float)
        self._norm_w = np.zeros(len(self.mc_arr), dtype=np.float)

        # Loop over all selected events per src and calc the raw_flux per src
        for src_idx in np.unique(self.mc_arr["src_enum"]):
            src_mask = (self.mc_arr["src_enum"] == src_idx)

            # Finalize event weight calculation and save in mc_arr
            trueEi = self.mc_arr["trueE"][src_mask]
            omegai = self._omega[src_idx]
            self.mc_arr["ow"][src_mask] *= trueEi**(-self.gamma) / omegai

            # Raw flux per source
            self._raw_flux_per_src[src_idx] = np.sum(
                self.mc_arr["ow"][src_mask], dtype=np.float)

            # Normalized weights per source to flux per source to obtain a
            # weight normalized for each src on its own.
            self._norm_w[src_mask] = (self.mc_arr["ow"][src_mask] /
                                      self._raw_flux_per_src[src_idx])
            # Now multiply with normalized src weights to get sampling weight
            # for all events in the selected sample.
            # This automatically samples events at srcs with higher weights.
            self._norm_w[src_mask] *= self._src_norm_w[src_idx]

            # Double-check if no weight is dominating the sample
            if self._norm_w[src_mask].max() > 0.1:
                logger.warn("Warning: Maximal weight exceeds 10%: " +
                            "{0:7.2%}".format(self._norm_w[src_mask].max()))

        # Calc total inserted flux for all srcs by weighting with src weights
        self._raw_flux = np.sum(self._src_norm_w * self._raw_flux_per_src)
        return

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, val):
        self._gamma = float(val)
        if self.mc is not None:
            self._src_dec_weight_spline()
        else:
            print("No MC setup for effA. Use fill() to do so.")
        return

    # Public methods
    def src_dec_weights(self, src_dec, **params):
        """
        Calculates src detector weights from the precaluclated src weight
        spline for given declinations `src_dec`. This is dependent on the
        set vale for the spectral index `gamma`.

        Same function as in `ps_model.py`.

        Parameters
        ----------
        src_dec : array
            Array of src declinations in radian: [-pi/2, pi/2]

        Returns
        -------
        src_dec_w : array
            Weights for each given src declination. For declinations outside
            `sinDec_range` the weights are set to zero.
        """
        if self._spl_src_dec_weights is None:
            raise ValueError("Need to fill() with MC before effA calculation.")
        src_dec = np.atleast_1d(src_dec)
        src_dec_w = self._spl_src_dec_weights(np.sin(src_dec))
        invalid = ((np.sin(src_dec) < self.sinDec_range[0]) |
                   (np.sin(src_dec) > self.sinDec_range[1]))
        src_dec_w[invalid] = 0.

        return src_dec_w

    # def effA(self, dec):
    #     if self._spl_effA is None:
    #         raise ValueError("Need to fill() with MC before effA calculation.")
    #     """Vectorized version for multiple dec"""
    #     dec = np.atleast_1d(dec)
    #     effA = self._spl_effA(np.sin(dec))
    #     invalid = ((np.sin(dec) < self.sinDec_bins[0]) |
    #                (np.sin(dec) > self.sinDec_bins[-1]))
    #     effA[invalid] = 0.
    #     return effA

    def fill(self, mc, livetime, src_array, src_priors=None):
        # This only saves variables as class attributes.
        # Everytime we sample, we select events beforehand
        if isinstance(mc, dict) ^ isinstance(livetime, dict):
            raise ValueError("mc and livetime not compatible")

        if not isinstance(mc, dict):
            mc = {-1: mc}
            livetime = {-1: livetime}

        self.mc = mc
        self.livetime = livetime

        # Just print some info about MC input and check if sinDec is available
        print("MC input info:")
        for key in self.mc.iterkeys():
            print("  Sample {}: {} events with livetime {} d.".format(
                key, len(mc[key]), livetime[key]))
            if "sinDec" not in self.mc[key].dtype.fields:
                self.mc[key] = np.lib.recfunctions.append_fields(
                    self.mc[key], "sinDec", np.sin(self.mc[key]["dec"]),
                    dtypes=np.float, usemask=False)

        # Set sources
        if (("ra" not in src_array.dtype.names) or
                ("dec" not in src_array.dtype.names) or
                ("src_w" not in src_array.dtype.names)):
            raise ValueError("Need fields 'ra', 'dec', 'src_w' in src_array.")

        # Make sure we are using 1D arrays
        src_ra = np.atleast_1d(src_array["ra"])
        src_dec = np.atleast_1d(src_array["dec"])
        src_w = np.atleast_1d(src_array["src_w"])

        # Check if coordinates are valid equatorial coordinates in radians
        if np.any(src_ra < 0) or np.any(src_ra > 2 * np.pi):
            raise ValueError("RA value(s) not valid equatorial coordinates")
        if (np.any(src_dec < -np.pi / 2.) or np.any(src_dec > +np.pi / 2.)):
            raise ValueError("DEC value(s) not valid equatorial coordinates")

        # Zero or smaller weights make no sense
        if np.any(src_w <= 0):
            raise ValueError("Invalid source weight(s) <= 0 detected.")
        # End of sanity checks

        self._nsrcs = len(src_array)

        # Save as a recarray class variable
        self._src = np.empty((self._nsrcs, ), dtype=[
            ("ra", np.float), ("dec", np.float), ("src_w", np.float)])
        self._src["ra"] = src_ra
        self._src["dec"] = src_dec
        self._src["src_w"] = src_w

        # Setup prior maps if given, first check that healpy maps are valid
        if src_priors is not None:
            # Priors are arrays of map arrays
            src_priors = np.atleast_2d(src_priors)
            if not hp.maptype(src_priors) == len(self._src):
                raise ValueError("Healpy map priors must match number of srcs")
            else:
                self._src_priors = src_priors
                self._NSIDE = hp.get_nside(src_priors[0])

        # Calculate src detector weight spline once for current MCs and gamma
        self._effA()

        return

    def sample(self, src_ra, mean_mu, poisson=True, ret_src_dir=False):
        """External src_ra not needed but kept for usability in psLLH.py
        If ret_src_dir is true, it returns src ras/decs per trial. This is
        needed for BG trials, if the src positions are fitted too."""
        # Generate event numbers using poissonian events
        while True:
            num = (self.random.poisson(mean_mu)
                   if poisson else int(np.around(mean_mu)))

            logger.debug(("Generated number of sources: {0:3d} " +
                          "of mean {1:5.1f} sources").format(num, mean_mu))

            # If no events should be sampled, return nothing
            if num < 1:
                yield num, None
                continue

            # If we have src priors, sample new src positions from each prior
            if self._src_priors is not None:
                src_idx = np.zeros(len(self._src_priors), dtype=int) - 1
                for i, prior in enumerate(self._src_priors):
                    src_idx[i] = amp_hp.healpy_rejection_sampler(prior, n=1)[0]
                # Get equatorial coordinates for the new src positions
                src_th, src_phi = hp.pix2ang(self._NSIDE, src_idx)
                src_dec, src_ra = amp_hp.ThetaPhiToDecRa(src_th, src_phi)
                src_dec = np.atleast_1d(src_dec)
                src_ra = np.atleast_1d(src_ra)
            else:
                # Otherwise use static given src positions for each trial
                src_ra = self._src["ra"]
                src_dec = self._src["dec"]

            # Now setup dec bands and select events for the new src positions
            self._setup(src_dec)
            self._select_events(src_dec)

            # Sample num events from all selected events with the global weight
            sam_idx = self.random.choice(self.mc_arr, size=num, p=self._norm_w)
            self.sam = sam_idx

            # Get the sample ids from the different MCs that were selected
            enums = np.unique(sam_idx["enum"])

            # Only one sample selected or given (-1), just return recarray
            if len(enums) == 1 and enums[0] < 0:
                # For the sample, get the source keys from which events were
                # selected
                _src_keys = self.mc_sel[enums[0]].keys()
                # Append events from all srcs where events were selected
                _sam_ev = []
                for i in _src_keys:
                    idx = sam_idx[sam_idx["src_enum"] == i]["idx"]
                    if len(idx) == 0:
                        continue
                    _sam_ev.append(rotate_struct(
                        self.mc_sel[enums[0]][i][idx], src_ra[i], src_dec[i]))

                # Concat split src arrays to single output array
                sam_ev = np.array([], dtype=_sam_ev[0].dtype)
                for _sam_ev_i in _sam_ev:
                    sam_ev = np.append(sam_ev, _sam_ev_i)

                if ret_src_dir:
                    yield num, sam_ev, src_ra, src_dec
                else:
                    yield num, sam_ev
                continue

            # Otherwise return samples in dict with MC key they belong to
            sam_ev = dict()
            for enum in enums:
                _src_keys = self.mc_sel[enum].keys()
                # For each sample get MC from every src selected
                _sam_ev = []
                for i in _src_keys:
                    enum_mask = (sam_idx["enum"] == enum)
                    src_mask = (sam_idx[enum_mask]["src_enum"] == i)
                    idx = sam_idx[enum_mask][src_mask]["idx"]
                    if len(idx) == 0:
                        continue
                    _sam_ev.append(rotate_struct(
                        self.mc_sel[enum][i][idx], src_ra[i], src_dec[i]))

                # Concat split src arrays to single output array per sample
                sam_ev_i = np.array([], dtype=_sam_ev[0].dtype)
                for _sam_ev_i in _sam_ev:
                    sam_ev_i = np.append(sam_ev_i, _sam_ev_i)
                sam_ev[enum] = sam_ev_i

            if ret_src_dir:
                yield num, sam_ev, src_ra, src_dec
            else:
                yield num, sam_ev
