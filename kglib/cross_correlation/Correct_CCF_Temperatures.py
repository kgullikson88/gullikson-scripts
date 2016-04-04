"""
This file contains a couple functions for taking a dataframe of measured temperatures
and associated starnames/instruments, and returning a corrected temperature with error bar.
The correction needed is determined in the iPython notebook 'CheckCCFSystematics'.
"""


import pandas as pd



def get_real_temperature_newmethod(df, addmode='simple'):
    """
    See docstring for get_real_temperature. This function is only for legacy support.

    """
    return get_real_temperature(df, addmode=addmode)


def get_real_temperature(df, addmode='simple'):
    """
    Given a dataframe of observations, find the actual temperature and uncertainty for each star

    Parameters:
    ===========
    - df:               pandas DataFrame
                        The input dataframe. Must have the following keys:
                            - 'Star'
                            - '[Fe/H]'
                            - 'vsini'  (which is the vsini of the secondary star)
                            - 'Instrument'
                            - 'Temperature' (which is the measured temperature)

    - addmode:          string
                        The way the individual order CCFs were co-added

    Returns:
    ========
    corrected:          pandas DataFrame
                        A dataframe with the corrected temperature, and its upper and lower errors.
    """
    from kglib.utils import HDF5_Helpers
    hdf_interface = HDF5_Helpers.Full_CCF_Interface()

    # Group by the star name.
    star_groups = df.groupby('Star')
    starnames = star_groups.groups.keys()
    metal = []
    vsini = []
    corrected_list = []
    for starname in starnames:
        # Get the measured temperature for each observation
        star_df = star_groups.get_group(starname)
        m_list = []
        for _, r in star_df.iterrows():
            m_list.append(
                hdf_interface.get_measured_temperature(r['Star'], r['Date'], r['Temperature'], r['Instrument']))
        measurements = pd.concat(m_list, ignore_index=True)

        # Convert the measurements to actual temperatures
        corrected = HDF5_Helpers.convert_measured_to_actual(measurements.copy())
        corrected_list.append(corrected)

    corrected = pd.concat(corrected_list, ignore_index=True)

    def to_latex(x, uperr, lowerr):
        return '{:.0f}^{{+{:.0f}}}_{{-{:.0f}}}'.format(x, uperr, lowerr)

    temperature_latex = corrected.apply(lambda r: to_latex(r['Corrected_Temperature'],
                                                           r['Scaled_T_uperr'],
                                                           r['Scaled_T_lowerr']),
                                        axis=1)

    return pd.DataFrame(data={'Star': starnames, '[Fe/H]': metal, 'vsini': vsini,
                              'Temperature_tex': temperature_latex,
                              'Temperature': corrected['Corrected_Temperature'],
                              'temperature_lowerr': corrected['Scaled_T_lowerr'],
                              'temperature_uperr': corrected['Scaled_T_uperr']})
