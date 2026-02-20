def cross_match_survey_circle(
    source_tmin,
    source_tmax,
    source_ra,
    source_dec,
    survey_t,
    survey_ra,
    survey_dec,
    survey_fov,
    chunk_time=6 * 30,
):
    """
    Cross match between a survey and a list of transients.

    Search through the times (survey_t) and positions (survey_ra, survey_dec) of
    a survey imaging campaign and check which ones would overlap a transient
    source. The transient is assumed to be a point source (at source_ra,
    source_dec), and only visible between source_tmin and source_tmax. The
    observable region is assumed to be circular with diameter survey_fov.

    This may be sufficient for simulation purposes, or may just be a good
    initial trimming of a massive survey and transient database into a more
    manageable size.

    The result is a list of lists. For every source a list of indices is given
    which are the indices of the corresponding observations in the survey.

    Parameters
    ----------
    source_tmin : np.array
        Start time for the transient, earliest time it is visible. (MJD)
    source_tmax : np.array
        End time for the transient, last time it is visible. (MJD)
    source_ra : np.array
        Right Ascension coordinate for transient. (deg)
    source_dec : np.array
        Declination coordinate for transient. (deg)
    survey_t : np.array
        Image time, assumed essentially instantaneous. (MJD)
    survey_ra : np.array
        Right Ascension coordinate of center of image. (deg)
    survey_dec : np.array
        Declination coordinate of center of image. (deg)
    chunk_time : float
        The sources and survey are chunked into blocks of this much time.
        Trimming on a single axis such as time is very fast and can massively
        reduce the number of cross matches to make, speeding up the whole
        process if chosen correctly. The default chunk_time is chosen for a
        source transient that has a timescale on the order of a few months.

    Returns
    -------
    match_indices : list[list]
        Matches between sources and survey images. List with entry for every
        source (same order), with elements that are lists of the indices of
        matching survey observations.
    """
