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


def get_fivebyfive(file, latitude, longitude, rrs_wavelengths):
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
        rrs_data = (
            ds_data["Rrs"].isel(
                number_of_lines=slice(line_start, line_end),
                pixels_per_line=slice(pixel_start, pixel_end),
            ).values
        )
        flags_data = (
            ds_data["l2_flags"].isel(
                number_of_lines=slice(line_start, line_end),
                pixels_per_line=slice(pixel_start, pixel_end),
            ).values
        )

    # Calculate the bitwise OR of all flags in EXCLUSION_FLAGS to get a mask
    exclude_mask = sum(L2_FLAGS[flag] for flag in EXCLUSION_FLAGS)

    # Create a boolean mask
    # True means the flag value does not contain any of the EXCLUSION_FLAGS
    valid_mask = np.bitwise_and(flags_data, exclude_mask) == 0

    # Get stats and averages
    if valid_mask.any():
        rrs_valid = rrs_data[valid_mask]
        rrs_std_initial = np.std(rrs_valid, axis=0)
        rrs_mean_initial = np.mean(rrs_valid, axis=0)

        # Exclude spectra > 1.5 stdevs away
        std_mask = np.all(
            np.abs(rrs_valid - rrs_mean_initial) <= 1.5 * rrs_std_initial,
            axis=1
        )
        rrs_std = np.std(rrs_valid[std_mask], axis=0)
        rrs_mean = np.mean(rrs_valid[std_mask], axis=0).flatten()

        # Matchup criteria uses cv as median of 405-570nm
        rrs_cv = rrs_std / rrs_mean
        rrs_cv_median = np.median(
            rrs_cv[(rrs_wavelengths >= 405) & (rrs_wavelengths <= 570)]
        )
    else:
        rrs_cv_median = np.nan
        rrs_mean = np.nan * np.empty_like(rrs_wavelengths)

    # Put in dictionary of the row
    row = {
        "sat_datetime": pd.to_datetime(
            file.granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"],
            utc=0
        ),
        "sat_cv": rrs_cv_median,
        "sat_latitude": sat_lat[center_line, center_pixel],
        "sat_longitude": sat_lon[center_line, center_pixel],
        "sat_pixel_valid": np.sum(valid_mask),
    }

    # Add mean spectra to the row dictionary
    for wavelength, mean_value in zip(rrs_wavelengths, rrs_mean):
        key = f"sat_rrs{int(wavelength)}"
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
    time_bounds = (f"{start_date}T00:00:00", f"{end_date}T23:59:59")

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

    # Pull out Rrs wavelengths for easier processing
    with xr.open_dataset(files[0], group="sensor_band_parameters") as ds_bands:
        rrs_wavelengths = ds_bands["wavelength_3d"].values

    # Loop through files and process
    sat_rows = []
    for idx, file in enumerate(files):
        granule_date = pd.to_datetime(
            file.granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
        )
        print(f"Running Granule: {granule_date}")
        row = get_fivebyfive(file, latitude, longitude, rrs_wavelengths)
        sat_rows.append(row)

    return pd.DataFrame(sat_rows)


##---------------------------------------------------------------------------##
#                              Matchup Utilities                              #
##---------------------------------------------------------------------------##


def match_data(
    df_sat,
    df_field,
    cv_max=0.15,
    senz_max=60.0,
    min_percent_valid=55.0,
    max_time_diff=180,
    std_max=1.5,
):
    """Create matchup dataframe based on selection criteria.

    Parameters
    ----------
    df_sat : pandas dataframe
        Satellite data from flat validation file.
    df_field : pandas dataframe
        Field data from flat validation file.
    cv_max : float, default 0.15
        Maximum coefficient of variation (stdev/mean) for sat data.
    senz_max : float, default 60.0
        Maximum sensor zenith for sat data.
    min_percent_valid : float, default 55.0
        Minimum percentage of valid satellite pixels.
    max_time_diff : int, default 180
        Maximum time difference (minutes) between sat and field matchup.
    std_max : float, default 1.5
        If multiple valid field matchups, select within std_max stdevs of mean.

    Returns
    -------
    pandas dataframe of matchups for product

    Notes
    -----
    This is hard-coded to match on Rrs for the demo. For other products, take
    out the cv parameter and make the row product column search more generic.
    """
    # Setup
    time_window = pd.Timedelta(minutes=max_time_diff)
    df_match_list = []

    # Filter Field data based on Solar Zenith
    df_field_filtered = df_field[df_field["field_solar_zenith"] <= senz_max]

    # Filter satellite data based on cv threshold
    df_sat_filtered = df_sat[df_sat["sat_cv"] <= cv_max]

    # Filter satellite data based on percent good pixels
    df_sat_filtered = df_sat_filtered[
        df_sat_filtered["sat_pixel_valid"] >= min_percent_valid * 25 / 100
    ]

    for _, sat_row in df_sat_filtered.iterrows():
        # Filter field data based on time difference and coordinates
        time_diff = abs(
            df_field_filtered["field_datetime"] - sat_row["sat_datetime"]
            )
        time_mask = time_diff <= time_window
        lat_mask = 0.2 >= abs(
            df_field_filtered["field_latitude"] - sat_row["sat_latitude"]
        )
        lon_mask = 0.2 >= abs(
            df_field_filtered["field_longitude"] - sat_row["sat_longitude"]
        )
        field_matches = df_field_filtered[time_mask & lat_mask & lon_mask]

        if field_matches.shape[0] > 5:
            # Filter by Standard Deviation for rrs columns
            rrs_cols = [
                col for col in field_matches.columns
                if col.startswith("field_rrs")
                and int(col.rsplit("_rrs")[1]) >= 400
                and int(col.rsplit("_rrs")[1]) <= 700
            ]
            if rrs_cols:
                mean_spectra = field_matches[rrs_cols].mean(axis=0)
                std_spectra = field_matches[rrs_cols].std(axis=0)
                within_std = (
                    abs(field_matches[rrs_cols] - mean_spectra) <= std_max * std_spectra
                )
                field_matches = field_matches[within_std.all(axis=1)]

        if not field_matches.empty:
            # Select the best match based on time delta
            time_diff = abs(
                field_matches["field_datetime"] - sat_row["sat_datetime"]
                )
            best_match = field_matches.loc[time_diff.idxmin()]
            df_match_list.append({**best_match.to_dict(), **sat_row.to_dict()})

    df_match = pd.DataFrame(df_match_list)
    return df_match