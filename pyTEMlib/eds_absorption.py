density_of_mixture

def _density_of_mixture(weight_percent, elements, mean="harmonic"):
    """Calculate the density a mixture of elements.

    The density of the elements is retrieved from an internal database. The
    calculation is only valid if there is no interaction between the
    components.

    Parameters
    ----------
    weight_percent : array
        A list of weight percent for the different elements. If the total
        is not equal to 100, each weight percent is divided by the sum
        of the list (normalization).
    elements : list of str
        A list of element symbols, e.g. ['Al', 'Zn']
    mean : ``'harmonic'`` or ``'weighted'``
        The type of mean use to estimate the density.
        Default is ``'harmonic'``.

    Returns
    -------
    density : numpy.ndarray of float
        The density in g/cm3.

    Examples
    --------
    Calculate the density of modern bronze given its weight percent:

    >>> exspy.material.density_of_mixture([88, 12],['Cu', 'Sn'])
    8.6903187973131466

    """
    if len(elements) != len(weight_percent):
        raise ValueError(
            "The number of elements must match the size of the first axis"
            "of weight_percent."
        )
    densities = np.array(
        [
            elements_db[element]["Physical_properties"]["density (g/cm^3)"]
            for element in elements
        ]
    )
    sum_densities = np.zeros_like(weight_percent, dtype="float")
    try:
        if mean == "harmonic":
            for i, weight in enumerate(weight_percent):
                sum_densities[i] = weight / densities[i]
            sum_densities = sum_densities.sum(axis=0)
            density = np.sum(weight_percent, axis=0) / sum_densities
            return np.where(sum_densities == 0.0, 0.0, density)
        elif mean == "weighted":
            for i, weight in enumerate(weight_percent):
                sum_densities[i] = weight * densities[i]
            sum_densities = sum_densities.sum(axis=0)
            sum_weight = np.sum(weight_percent, axis=0)
            density = sum_densities / sum_weight
            return np.where(sum_weight == 0.0, 0.0, density)
    except TypeError:
        raise ValueError(
            "The density of one of the elements is unknown (Probably At or Fr)."
        )


def density_of_mixture(weight_percent, elements="auto", mean="harmonic"):
    """Calculate the density of a mixture of elements.

    The density of the elements is retrieved from an internal database. The
    calculation is only valid if there is no interaction between the
    components.

    Parameters
    ----------
    weight_percent : list of float or list of signals
        A list of weight percent for the different elements. If the total
        is not equal to 100, each weight percent is divided by the sum
        of the list (normalization).
    elements : list of str
        A list of element symbols, e.g. ['Al', 'Zn']. If elements is 'auto',
        take the elements in en each signal metadata of the weight_percent
        list.
    mean : ``'harmonic'`` or ``'weighted'``
        The type of mean use to estimate the density.
        Default is ``'harmonic'``.

    Returns
    -------
    density : numpy.ndarray of float or hyperspy.api.signals.BaseSignal
        The density in g/cm3.

    Examples
    --------
    Calculate the density of modern bronze given its weight percent:

    >>> exspy.material.density_of_mixture([88, 12],['Cu', 'Sn'])
    8.6903187973131466

    """
    from hyperspy.signals import BaseSignal

    elements = _elements_auto(weight_percent, elements)
    if isinstance(weight_percent[0], BaseSignal):
        density = weight_percent[0]._deepcopy_with_new_data(
            _density_of_mixture(stack(weight_percent).data, elements, mean=mean)
        )
        return density
    else:
        return _density_of_mixture(weight_percent, elements, mean=mean)


def mass_absorption_coefficient(element, energies):
    """
    Mass absorption coefficient (mu/rho) of a X-ray absorbed in a pure
    material.

    The mass absorption is retrieved from the database of Chantler et al. [*]_.

    Parameters
    ----------
    element: str
        The element symbol of the absorber, e.g. 'Al'.
    energies: float or list of float or str or list of str
        The energy or energies of the X-ray in keV, or the name of the X-rays,
        e.g. 'Al_Ka'.

    Return
    ------
    mass_absorption_coefficients : numpy.ndarray of float
        Mass absorption coefficient(s) in cm^2/g

    Examples
    --------
    >>> exspy.material.mass_absorption_coefficient(
    >>>     element='Al', energies=['C_Ka','Al_Ka'])
    array([ 26330.38933818,    372.02616732])

    See also
    --------
    exspy.material.mass_absorption_mixture

    References
    ----------
    .. [*] Chantler, C.T., Olsen, K., Dragoset, R.A., Kishore, A.R., Kotochigova,
       S.A., and Zucker, D.S. (2005), X-Ray Form Factor, Attenuation and
       Scattering Tables (version 2.1). https://dx.doi.org/10.18434/T4HS32
    """
    energies_db = np.array(ffast_mac[element].energies_keV)
    macs = np.array(ffast_mac[element].mass_absorption_coefficient_cm2g)
    energies = copy.copy(energies)
    if isinstance(energies, str):
        energies = utils_eds._get_energy_xray_line(energies)
    elif isinstance(energies, Iterable):
        for i, energy in enumerate(energies):
            if isinstance(energy, str):
                energies[i] = utils_eds._get_energy_xray_line(energy)
    index = np.searchsorted(energies_db, energies)
    mac_res = np.exp(
        np.log(macs[index - 1])
        + np.log(macs[index] / macs[index - 1])
        * (
            np.log(energies / energies_db[index - 1])
            / np.log(energies_db[index] / energies_db[index - 1])
        )
    )
    return np.nan_to_num(mac_res)


def _mass_absorption_mixture(weight_percent, elements, energies):
    """Calculate the mass absorption coefficient for X-ray absorbed in a
    mixture of elements.

    The mass absorption coefficient is calculated as a weighted mean of the
    weight percent and is retrieved from the database of Chantler et al. [*]_.

    Parameters
    ----------
    weight_percent: np.array
        The composition of the absorber(s) in weight percent. The first
        dimension of the matrix corresponds to the elements.
    elements: list of str
        The list of element symbol of the absorber, e.g. ['Al','Zn'].
    energies: float or list of float or str or list of str
        The energy or energies of the X-ray in keV, or the name of the X-rays,
        e.g. 'Al_Ka'.

    Examples
    --------
    >>> exspy.material.mass_absorption_mixture(
    >>>     elements=['Al','Zn'], weight_percent=[50,50], energies='Al_Ka')
    2587.4161643905127

    Return
    ------
    mass_absorption_coefficient : numpy.ndarray of float
        The mass absorption coefficient(s) in cm^2/g

    See also
    --------
    exspy.material.mass_absorption

    References
    ----------
    .. [*] Chantler, C.T., Olsen, K., Dragoset, R.A., Kishore, A.R., Kotochigova,
       S.A., and Zucker, D.S. (2005), X-Ray Form Factor, Attenuation and
       Scattering Tables (version 2.1). https://dx.doi.org/10.18434/T4HS32
    """
    if len(elements) != len(weight_percent):
        raise ValueError("Elements and weight_fraction should have the same length")
    if isinstance(weight_percent[0], Iterable):
        weight_fraction = np.array(weight_percent)
        weight_fraction /= np.sum(weight_fraction, 0)
        mac_res = np.zeros([len(energies)] + list(weight_fraction.shape[1:]))
        for element, weight in zip(elements, weight_fraction):
            mac_re = mass_absorption_coefficient(element, energies)
            mac_res += np.array([weight * ma for ma in mac_re])
        return mac_res
    else:
        mac_res = np.array(
            [mass_absorption_coefficient(el, energies) for el in elements]
        )
        mac_res = np.dot(weight_percent, mac_res) / np.sum(weight_percent, 0)
        return mac_res


def mass_absorption_mixture(weight_percent, elements="auto", energies="auto"):
    """Calculate the mass absorption coefficient for X-ray absorbed in a
    mixture of elements.

    The mass absorption coefficient is calculated as a weighted mean of the
    weight percent and is retrieved from the database of Chantler et al. [*]_.

    Parameters
    ----------
    weight_percent: list of float or list of signals
        The composition of the absorber(s) in weight percent. The first
        dimension of the matrix corresponds to the elements.
    elements: list of str or 'auto'
        The list of element symbol of the absorber, e.g. ['Al','Zn']. If
        elements is 'auto', take the elements in each signal metadata of the
        weight_percent list.
    energies: list of float or list of str or 'auto'
        The energy or energies of the X-ray in keV, or the name of the X-rays,
        e.g. 'Al_Ka'. If 'auto', take the lines in each signal metadata of the
        weight_percent list.

    Examples
    --------
    >>> exspy.material.mass_absorption_mixture(
    >>>     elements=['Al','Zn'], weight_percent=[50,50], energies='Al_Ka')
    2587.41616439

    Return
    ------
    mass_absorption_coefficient : numpy.ndarray of float or hyperspy.api.signals.BaseSignal
        The Mass absorption coefficient(s) of the mixture in cm^2/g

    See also
    --------
    exspy.material.mass_absorption_coefficient

    References
    ----------
    .. [*] Chantler, C.T., Olsen, K., Dragoset, R.A., Kishore, A.R., Kotochigova,
       S.A., and Zucker, D.S. (2005), X-Ray Form Factor, Attenuation and
       Scattering Tables (version 2.1). https://dx.doi.org/10.18434/T4HS32
    """
    from hyperspy.signals import BaseSignal

    elements = _elements_auto(weight_percent, elements)
    energies = _lines_auto(weight_percent, energies)
    if isinstance(weight_percent[0], BaseSignal):
        weight_per = np.array([wt.data for wt in weight_percent])
        mac_res = stack(
            [weight_percent[0].deepcopy()] * len(energies), show_progressbar=False
        )
        mac_res.data = _mass_absorption_mixture(weight_per, elements, energies)
        mac_res = mac_res.split()
        for i, energy in enumerate(energies):
            mac_res[i].metadata.set_item("Sample.xray_lines", ([energy]))
            mac_res[i].metadata.General.set_item(
                "title",
                "Absoprtion coeff of"
                " %s in %s" % (energy, mac_res[i].metadata.General.title),
            )
            if mac_res[i].metadata.has_item("Sample.elements"):
                del mac_res[i].metadata.Sample.elements
        return mac_res
    else:
        return _mass_absorption_mixture(weight_percent, elements, energies)
