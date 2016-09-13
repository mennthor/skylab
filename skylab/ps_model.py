
##############################################################################
## HealpyLLH variable defaults
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
    _cached_maps = _cached_maps


     # INTERNAL METHODS
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


    # GETTER / SETTER
    # Smoothed healpy maps are cached for every event beforehand to shorten
    # signal computation
    @property
    def cached_maps(self):
        return self._cached_maps
    @cached_maps.setter
    def cached_maps(self, maps):
        # This will throw a TypeError, if maps are not valid hp maps
        hp.maptype(maps)
        self._cached_maps = maps
        return


    # PUBLIC METHODS
    def signal(self, ev, mask):
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
        mask : bool array
            Selected events mask. This mask selects the correct cached map for
            the ith event in the given `ev` array.

        Returns
        --------
        S : array-like
            Spatial signal probability for each event in ev.
        """
        # Check if we have cached exp maps, should always be the case
        if self.cached_exp_maps is None:
            raise ValueError("We don't have cached maps, need to add sources"
            + " first using `psLLH.use_source()`")

        # Event array and mask must be fitting
        if len(ev) != len(mask):
            raise ValueError("ev array and mask do not fit each other.")

        # Select the correct maps for the input events
        maps = self._cached_maps[mask]

        # Get all pdf values: first get pixel indices for the events in ev
        NSIDE = hp.get_nside(maps[0])
        NPIX = hp.nside2npix(NSIDE)
        # Shift RA, DEC to healpy coordinates for proper use of pix indices
        th, phi = amp_hp.DecRaToThetaPhi(ev["dec"], ev["ra"])
        pixind = hp.ang2pix(NSIDE, th, phi)

        # For every event get the correct spatial pdf signal value.
        # Because the src maps were added with the weight beforehand, this
        # already is the stacked llh value for the signal
        S = [maps[i][ind] for i, ind in enumerate(pixind)]

        return np.array(S)


    def reset_map_cache(self):
        r"""
        Reset all cached maps. Resetting the map cache has its own function
        to not interfere with the usual reset, which is for selecting events
        and weights.
        """
        self._cached_maps = _cached_maps
        return
