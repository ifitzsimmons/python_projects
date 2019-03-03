import argparse
import numpy as np
import pandas as pd
from pprint import pprint
import sys
import unittest

def _parse_args():
    parser = argparse.ArgumentParser(
        description=('This module returns lists of tuples that '
                     'describe similarities of two locations '
                     'based on the distribution of races.'))

    parser.add_argument('--top-results', type=int, default=5,
                        help='Number of results displayed')

    return parser

def filter_by_race(df):
    """Filter DataFrame
    
    Parameters
    ----------
    df: pandas.DataFrame
        Population data that must include columns with  
        the following races
        ['White', 'Black', 'Hispanic', 'Asian',
        'American_Indian_or_Alaska_Native',
        'Native_Hawaiian_or_Other_Pacific_Islander', 'Two_Or_More_Races']

    Returns
    -------
    races_df: pandas.DataFrame
        Create a new DataFrame that only contains the population by race

    """
    
    races = [
        'White', 'Black', 'Hispanic', 'Asian', 
        'American_Indian_or_Alaska_Native', 
        'Native_Hawaiian_or_Other_Pacific_Islander', 
        'Two_Or_More_Races'
        ]

    races_df = df[races].copy().fillna(0)
    
    return races_df

def get_cos_sim(array_1, array_2):
    """Calculate cosine similarity of two numpy arrays
    
    Parameters
    ----------
        array_1:  np.ndarray, shape [n_locations, m_races_considered]
        array_2:  np.ndarray, shape [n_locations, m_races_considered]

    Returns
    -------
    cos_sim: float
        cosine similarity of two arrays

    """
    cos_sim = array_1.dot(array_2) / (np.linalg.norm(array_1) * np.linalg.norm(array_2))

    if np.isnan(cos_sim):
        cos_sim = 0.0

    return cos_sim

def get_state_similarities(dem_df):
    """Find similarity of race distribution between states
    
    Parameters
    ----------
    dem_df: pandas.DataFrame
        DataFrame with state/locations as index and 
        population by race in each column

    Returns
    -------
    sorted_sim: list
        sorted list of tuples containing Location 1, Location 2, and 
        their cosine similarity of the population distribution. Sorted 
        from most similar to least similar

    """
    
    demographic_similarities = []

    for i, row in enumerate(dem_df.iterrows()):
        # Unpack row by index, rest of row
        location_1, demographics_1 = row
        
        dem_array = np.array(demographics_1)
        
        for j, comparison_row in enumerate(dem_df.iterrows()):
            # j>i eliminates possibility of duplicate entries in dem_list
            if j > i:
                comp_location, comp_dem = comparison_row
                comp_array = np.array(comp_dem)

                # cosine similarity = (dot product of two arrays)/(product of their magnitudes)
                cos_sim = get_cos_sim(dem_array, comp_array)
                demographic_similarities.append((location_1, comp_location, cos_sim))

    sorted_sim = sorted(demographic_similarities, key=lambda x: x[2], reverse=True)
    return sorted_sim

def get_sim_info(similarities, top=5):
    """Find most and least common states by race distribution
    
    Parameters
    ----------
    similarities: list
        Reverse sorted List containing two states and their similarity sorted
        by similarity from most to least similar
    top: int
        Number of results returned. 

    Returns
    -------
    most_similar, least_similar: tuple(list, list)
        most_similar - Sorted list of tuples containing Location 1 and Location 2. Sorted 
        from most similar to least similar
        least_similar - Sorted list of tuples containing Location 1 and Location 2. Sorted 
        from least similar to most similar

    """
    
    most_similar = [(item[0], item[1]) for item in similarities[:top]]
    least_similar = [(item[0], item[1]) for item in reversed(similarities[-top:])]
    return most_similar, least_similar


if __name__ == "__main__":
    top_results = _parse_args().parse_args().top_results

    path = 'data/demographics.csv'
    data = pd.read_csv(path)
    data = data.set_index('Location')
    data['total'] = data['Male'] + data['Female']
    filtered = filter_by_race(data)
    similarities =  get_state_similarities(filtered)
    most, least = get_sim_info(similarities, top_results)

    print(f'These are the {top_results} most similar locations based on race distribution (from most to least similar).')
    pprint(most)
    print('\n')
    print(f'These are the {top_results} least similar locations based on race distribution (from least to most similar).')
    pprint(least)
    