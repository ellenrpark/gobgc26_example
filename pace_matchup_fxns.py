"""Helper functions for PACE Hackweek Validation Tutorial.

Authors:
    James Allen and Anna Windle

Source: https://pacehackweek.github.io/pace-2025/presentations/notebooks/satellite_insitu_matchups.html

Modified by Ellen Park for other PACE data products
"""

import datetime
import os
import re
from pathlib import Path

import earthaccess
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.ticker import FuncFormatter
from scipy import odr, stats

# Satellite Matchup Constants
# Short names for earthaccess lookup
SAT_LOOKUP = {
    "PACE_AOP": "PACE_OCI_L2_AOP",
    "PACE_IOP": "PACE_OCI_L2_IOP",
    "PACE_BGC": "PACE_OCI_L2_BGC",
    "AQUA": "MODISA_L2_OC",
    "TERRA": "MODIST_L2_OC",
    "NOAA-20": "VIIRSJ1_L2_OC",
    "NOAA-21": "VIIRSJ2_L2_OC",
    "SUOMI-NPP": "VIIRSN_L2_OC",
}

# List l2 flags, then build them into a dict
l2_flags_list = [
    "ATMFAIL",
    "LAND",
    "PRODWARN",
    "HIGLINT",
    "HILT",
    "HISATZEN",
    "COASTZ",
    "SPARE",
    "STRAYLIGHT",
    "CLDICE",
    "COCCOLITH",
    "TURBIDW",
    "HISOLZEN",
    "SPARE",
    "LOWLW",
    "CHLFAIL",
    "NAVWARN",
    "ABSAER",
    "SPARE",
    "MAXAERITER",
    "MODGLINT",
    "CHLWARN",
    "ATMWARN",
    "SPARE",
    "SEAICE",
    "NAVFAIL",
    "FILTER",
    "SPARE",
    "BOWTIEDEL",
    "HIPOL",
    "PRODFAIL",
    "SPARE",
]
L2_FLAGS = {flag: 1 << idx for idx, flag in enumerate(l2_flags_list)}

# Bailey and Werdell 2006 exclusion criteria
EXCLUSION_FLAGS = [
    "LAND",
    "HIGLINT",
    "HILT",
    "STRAYLIGHT",
    "CLDICE",
    "ATMFAIL",
    "LOWLW",
    "FILTER",
    "NAVFAIL",
    "NAVWARN",
]

# L2 Mask default
EXCLUSION_FLAGS = ["LAND", "HILT", "STRAYLIGHT", "CLDICE"]

##---------------------------------------------------------------------------##
#                             Satellite Utilities                             #
##---------------------------------------------------------------------------##


def parse_quality_flags(flag_value):
    """Parse bitwise flag into a list of flag names.

    Parameters
    ----------
    flag_value : int
        The integer representing the combined bitwise quality flags.

    Returns
    -------
    list of str
        List of flag names that are set in the flag_value.

    """
    return [
        flag_name for flag_name, value in L2_FLAGS.items()
        if (flag_value & value) != 0
    ]


def get_fivebyfive(file, latitude, longitude, sat, sat_variables):
    """Get stats on 5x5 box around station coordinates of a satellite granule.

    This checks l2flags and runs statistics on valid pixels and returns their
    valid count, the coefficient of variance (cv), and the Rrs values.

    Parameters
    ----------
    file : earthaccess granule object
        Satellite granule from earthaccess.
    latitude : float
        In decimal degrees for Aeronet-OC site for matchups
    longitude : float
        In decimal degrees (negative West) for Aeronet-OC site for matchups
    rrs_wavelengths ; numpy array
        Rrs wavelengths (from wavelength_3d for OCI)

    Returns
    -------
    dict
        A dictionary of the processed 5x5 box with:
            - "sat_datetime": pd.datetime
                Datetime of the overall granule start time
            - "sat_cv": float
                Median coefficient of variation of Rrs(405nm - 570nm)
            - "sat_latitude": float
                Latitude of center pixel
            - "sat_longitude": float
                Longitude of center pixel
            - "sat_pixel_valid": float
                Number of valid pixels in 5x5 box based on l2 flags

    Notes
    -----
    This is set to use just Rrs data for the demo. As an exercise, make this
    function more generalized by adding an input for the desired product and
    removing the wavelength dependency (if not needed) as well as the cv
    calculation. This will also require refactoring the `match_data` function.
    """
    with xr.open_dataset(file, group="navigation_data") as ds_nav:
        sat_lat = ds_nav["latitude"].values
        sat_lon = ds_nav["longitude"].values

    # Calculate the Euclidean distance for 2D lat/lon arrays
    distances = np.sqrt((sat_lat - latitude) ** 2 + (sat_lon - longitude) ** 2)

    # Find the index of the minimum distance
    # Dimensions are (lines, pixels)
    min_dist_idx = np.unravel_index(np.argmin(distances), distances.shape)
    center_line, center_pixel = min_dist_idx

    # Get indices for a 5x5 box around the center pixel
    line_start = max(center_line - 2, 0)
    line_end = min(center_line + 2 + 1, sat_lat.shape[0])
    pixel_start = max(center_pixel - 2, 0)
    pixel_end = min(center_pixel + 2 + 1, sat_lat.shape[1])

    # Extract the data
    # NOTE: This is hard-coded to Rrs from an L2 AOP file.
    with xr.open_dataset(file, group="geophysical_data") as ds_data:
        
        if 'AOP' in sat:
            sat_data = (ds_data["Rrs"].isel(
                    number_of_lines=slice(line_start, line_end),
                    pixels_per_line=slice(pixel_start, pixel_end)).values)
            
        elif 'BGC' in sat:

            # Subset pixels
            sat_data = ds_data.isel(
                    number_of_lines=slice(line_start, line_end),
                    pixels_per_line=slice(pixel_start, pixel_end))

            x = sat_data[sat_variables].to_array().values
            sat_data = (np.moveaxis(x, 0, -1))
            
        flags_data = (
            ds_data["l2_flags"].isel(
                number_of_lines=slice(line_start, line_end),
                pixels_per_line=slice(pixel_start, pixel_end),
            ).values)

    # Calculate the bitwise OR of all flags in EXCLUSION_FLAGS to get a mask
    exclude_mask = sum(L2_FLAGS[flag] for flag in EXCLUSION_FLAGS)

    # Create a boolean mask
    # True means the flag value does not contain any of the EXCLUSION_FLAGS
    valid_mask = np.bitwise_and(flags_data, exclude_mask) == 0

    # Get stats and averages
    if valid_mask.any():
        
        valid = sat_data[valid_mask]
        std_initial = np.std(valid, axis=0)
        mean_initial = np.mean(valid, axis=0)

        # Exclude spectra > 1.5 stdevs away
        std_mask = np.all(
            np.abs(valid - mean_initial) <= 1.5 * std_initial,
            axis=1
        )
        sat_std = np.std(valid[std_mask], axis=0)
        sat_mean = np.mean(valid[std_mask], axis=0).flatten()

        # Matchup criteria uses cv as median of 405-570nm
        cv = sat_std / sat_mean

        if 'AOP' in sat:
            cv_median = np.median(cv[(sat_variables >= 405) & (sat_variables <= 570)])
        else:
            cv_median = np.nan
            
    else:
        cv_median = np.nan
        sat_mean = np.nan * np.empty_like(np.array(sat_variables),dtype='float64')

    # Put in dictionary of the row
    row = {
        "sat_datetime": pd.to_datetime(
            file.granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"],
            utc=0
        ),
        "sat_cv": cv_median,
        "sat_latitude": sat_lat[center_line, center_pixel],
        "sat_longitude": sat_lon[center_line, center_pixel],
        "sat_pixel_valid": np.sum(valid_mask),
    }

    # Add mean spectra to the row dictionary
    for wavelength, mean_value in zip(sat_variables, sat_mean):

        if 'AOP' in sat:
            key = f"sat_rrs{int(wavelength)}"
        elif 'BGC' in sat:
            key = 'sat_'+wavelength
        row[key] = mean_value

    return row

def get_sat_ts_matchups(
    start_date,
    end_date,
    latitude,
    longitude,
    sat="PACE_AOP",
    selected_dates=None
):
    """Make satellite timeseries of matchups from single station.

    Caution: If the date or coordinates aren't formatted correctly, it might
    pull a huge granule list and take forever to run. If it takes more than 45
    seconds to print the number of granules, just kill the process.

    Uses the earthaccess package. Defaults to the PACE OCI L2 IOP datasets,
    but other satellites can be used if they have a corresponding short_name
    in the SAT_LOOKUP dictionary.

    Workflow:
        1. Get list of matchup granules
        2. Loop through each file and:
            2a. Find closest pixel to station, extract 5x5 pixel box
            2b. Exclude pixels based on l2_flags
            2c. Filtered mean to get single spectra
            2d. Compute statistics and save data row
        3. Organize output pandas dataframe

    Parameters
    ----------
    start_date : datetime or str
        Beginning of Aeronet data to run.
    end_date : datetime or str, optional
        End of Aeronet data to run.
    latitude : float
        In decimal degrees for Aeronet-OC site for matchups
    longitude : float
        In decimal degrees (negative West) for Aeronet-OC site for matchups
    sat : str
        Name of satellite to search. Must be in SAT_LOOKUP dict constant.
    selected_dates : list of str, optional
        If given, only pull granules if the dates are in this list

    Returns
    -------
    pandas DataFrame object
        Flattened table of all satellite granule matchups.

    """
    # Look up short name from constants
    if sat not in SAT_LOOKUP.keys():
        raise ValueError(
            f"{sat} is not in the lookup dictionary. Available "
            f"sats are: {', '.join(SAT_LOOKUP)}"
        )
    short_name = SAT_LOOKUP[sat]

    # Format search parameters
    # time_bounds = (f"{start_date}T00:00:00", f"{end_date}T23:59:59")
    time_bounds = (start_date, end_date)
    
    # Run Earthaccess data search
    results = earthaccess.search_data(
        point=(longitude, latitude),
        temporal=time_bounds,
        short_name=short_name
    )
    if selected_dates is not None:
        filtered_results = [
            result
            for result in results
            if result["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"][:10]
            in selected_dates
        ]
        print(f"Filtered to {len(filtered_results)} Granules.")
        files = earthaccess.open(filtered_results)
    else:
        files = earthaccess.open(results)

    sat_rows = []
    
    if len(files) > 0:
        # Pull out Rrs wavelengths for easier processing

        if 'AOP' in sat:
            with xr.open_dataset(files[0], group="sensor_band_parameters") as ds_bands:
                sat_variables = ds_bands["wavelength_3d"].values
        elif 'BGC' in sat:
            with xr.open_dataset(files[0], group="geophysical_data") as ds_bands:
                sat_variables = list(ds_bands.variables)[:-1]
                                     
        # Loop through files and process
        
        for idx, file in enumerate(files):
            granule_date = pd.to_datetime(
                file.granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
            )
            print(f"Running Granule: {granule_date}")
            row = get_fivebyfive(file, latitude, longitude, sat,  sat_variables)
            sat_rows.append(row)
    else:
        print('None granules found.')

    return pd.DataFrame(sat_rows)