
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
"""

# python packages
from copy import deepcopy
from itertools import product
import logging

# scipy-project imports
import numpy as np
import scipy.interpolate
from scipy.stats import norm

# local package imports
from . import set_pars
from .utils import kernel_func

############################################################################
## HealpyLLH extra imports
import healpy as hp
# Status bar for long caching of healpy maps
from tqdm import tqdm
# My analysis tools
import anapymods.healpy as amp_hp
############################################################################

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

##############################################################################
## HealpyLLH variable defaults
_spatial_pdf_map = None
_cached_maps = None
##############################################################################
    The signal pdf gets modified according to the stacking ansatz described in
    `Astr.Phys.J. 636:680-684, 2006 <http://arxiv.org/abs/astro-ph/0507120>`_.
    The likelihood function from ClassicLLH is extended through

    .. math::

       \mathcal{L} = \prod_i\left(\frac{n_s}{N}\mathcal{S}^\mathrm{tot}
                     + \left(1-\frac{n_s}{N}\right)\mathcal{B}\right)

    with the stacked signal term

    .. math:: \mathcal{S}^{tot} = \frac{\sum W^j R^j S_i^j}{\sum W^j R^j}

    Spatial signal pdfs are described by healpy maps which gets folded with the
    event sigma to create a combined directional pdf for each event.
    """
    # Default values for HealpyLLH class. Can be overwritten in constructor
    # by setting the attribute as keyword argument
    _spatial_pdf_map = _spatial_pdf_map
    _cached_maps = _cached_maps

    # INTERNAL METHODS
    def _cache_maps(self, exp, mc, src):
        r"""
        Cache maps for fast signal calculation.

        exp, mc : record arrays
            Data, MC from psLLH.
        src : record array
            Record array containing the source information. Needed fields are
            sigma : array
                Valid healpy map containing the spatial source pdf.
            normw : Float
                Containing the normed total weight per soruce. The total
                weight is the theoretical and the detector weight the
                source.
        """
        # First make single source spatial pdf by adding weighted maps
        self._spatial_pdf_map = self._make_spatial_pdf_map(src)

        # Cache maps with every exp/mc event sigma
        print("Start caching exp and mc maps. This may take a while.")
        sigma = np.append(exp["sigma"], mc["sigma"])
        self.cached_maps = np.array(self._convolve_maps(
            self._spatial_pdf_map, sigma))
        print("Done caching {} exp and mc maps:".format(len(self.cached_maps)))
        print("  exp maps : {}\n  mc  maps : {}".format(len(exp), len(mc)))

        return

    def _make_spatial_pdf_map(self, src):
        r"""
        Wrapper to make a spatial src map pdf from maps and weights.

        Parameters
        ----------
        src : record array
            Record array containing the source information. Needed fields are
            sigma : array
                Valid healpy map containing the spatial source pdf.
            normw : Float
                Containing the normed total weight per soruce. The total
                weight is the theoretical and the detector weight the
                source.

        Returns
        -------
        spatial_pdf_map : healpy map
            Single map containing added weighted submaps and normed to area=1.
        """
        added_map = self._add_weighted_maps(src["sigma"], src["normw"])
        self._spatial_pdf_map = amp_hp.norm_healpy_map(added_map)
        return self._spatial_pdf_map

    def _convolve_maps(self, m, sigma):
        """
        This function does:
        1. Smooth the given map m with every given sigma (aka gaussian
           convolution). Progress is tracked with tqdm because this may take
           a while.
        2. After convolution make a pdf from every map
        3. Put everything in an array and return that
        """
        convolved_maps = np.array(
            [amp_hp.norm_healpy_map(
                hp.smoothing(m, sigma=sigma_evi, verbose=False)
                )
            for sigma_evi in tqdm(sigma)]
            )
        return convolved_maps

    def _add_weighted_maps(self, m, w):
        """
        Add weighted healpy maps. Lenght of maps m and weights w must match.
        Maps and weights both must be of shape (nsrcs, ).
        """
        # Make sure we have arrays
        m = np.atleast_1d(m)
        w = np.atleast_1d(w)
        if len(w) != len(m):
            raise ValueError("Lenghts of map and weight vector don't match.")
        return np.sum(m * w)


    # PROPERTIES for public variables using getters and setters
    @property
    def cached_maps(self):
        # Smoothed healpy maps are cached for every event beforehand to shorten
        # signal computation
        return self._cached_maps
    @cached_maps.setter
    def cached_maps(self, maps):
        # This will throw a TypeError, if maps are not valid hp maps
        hp.maptype(maps)
        self._cached_maps = maps
        return


    # PUBLIC METHODS
    def signal(self, ev):
        r"""
        Spatial probability of each event i coming from extended source j.
        For each event a combinded source map is created and the spatial
        signal values is the value of this maps at the events position.
        To ensure fast execution for every event, each map is folded with the
        events sigma beforehand in the `_select_events()` internal method in
        the `psLLH` class.

        Parameters
        -----------
        ev : structured array
            Event array, import information: dec, ra, sigma. Combined events
            from exp and mc, selected in the `psLLH._select_events()` internal
            method.
            Field `idx` are selected events indices. This array selects the
            correct cached map for the ith event in the given `ev` array.

        Returns
        --------
        S : array-like
            Spatial signal probability for each event in ev.
        """
        # Check if we have cached exp maps, should always be the case
        if self.cached_maps is None:
            raise ValueError("We don't have cached maps, need to add sources"
            + " first using `psLLH.use_source()`")

        # Get IDs
        ind = ev["idx"]

        # Select the correct maps for the input events
        maps = self.cached_maps[ind]

        # Get all pdf values: first get pixel indices for the events in ev
        NSIDE = hp.get_nside(maps[0])
        NPIX = hp.nside2npix(NSIDE)
        # Shift RA, DEC to healpy coordinates for proper use of pix indices
        th, phi = amp_hp.DecRaToThetaPhi(ev["dec"], ev["ra"])
        pixind = hp.ang2pix(NSIDE, th, phi)

        # For every event get the correct spatial pdf signal value.
        # Because the src maps were added with the weight beforehand, this
        # already is the stacked llh value for the signal
        S = [maps[i][k] for i, k in enumerate(pixind)]

        return np.array(S)


    def reset_map_cache(self):
        r"""
        Reset all cached maps. Resetting the map cache has its own function
        to not interfere with the usual reset, which is for selecting events
        and weights.
        """
        self._cached_maps = _cached_maps
        return
