import pytest
import numpy as np
import pandas as pd
import find_similar_states as fss

def test_filter_by_race():
    test_frame = pd.DataFrame({
        'White': [1, 2, 3, 4], 'Black': [5, 6, 7, 8], 'Hispanic': [9, 10, 11, 12], 
        'Asian': [12, 11, 10, 9], 
        'American_Indian_or_Alaska_Native': [8, 7, 6, 5], 
        'Native_Hawaiian_or_Other_Pacific_Islander': [4, 3, 2, 1], 
        'Two_Or_More_Races': [1, 3, 5, 7], 'male':[2, 4, 6, 8],
        'female': [11, 13, 17, 19]
    })

    expected = pd.DataFrame({
        'White': [1, 2, 3, 4], 'Black': [5, 6, 7, 8], 'Hispanic': [9, 10, 11, 12], 
        'Asian': [12, 11, 10, 9], 
        'American_Indian_or_Alaska_Native': [8, 7, 6, 5], 
        'Native_Hawaiian_or_Other_Pacific_Islander': [4, 3, 2, 1], 
        'Two_Or_More_Races': [1, 3, 5, 7],
    })

    result = fss.filter_by_race(test_frame)
    pd.testing.assert_frame_equal(result.sort_index(axis=1), expected.sort_index(axis=1))

def test_cos_sim():
    array_1 = np.array([1, 0, 0])
    array_2 = np.array([1, 2, 2])

    expected = 1.0/3
    result = fss.get_cos_sim(array_1, array_2)
    assert result == pytest.approx(expected)

def test_get_state_similarities():
    test_frame = pd.DataFrame({
        'White': [1, 1, 10, 4], 'Black': [1, 1, 20, 4], 'Hispanic': [0, 0, 50, 1]
    })
    indices = ['A', 'B', 'C', 'D']
    test_frame.index = indices
    
    array_dict = {}
    for idx in indices:
        array_dict[idx] = np.array(test_frame.loc[idx])

    expected = [
        ('A', 'B', fss.get_cos_sim(array_dict['A'], array_dict['B'])),
        ('A', 'D', fss.get_cos_sim(array_dict['A'], array_dict['D'])),
        ('B', 'D', fss.get_cos_sim(array_dict['B'], array_dict['D'])),
        ('C', 'D', fss.get_cos_sim(array_dict['C'], array_dict['D'])),
        ('A', 'C', fss.get_cos_sim(array_dict['A'], array_dict['C'])),
        ('B', 'C', fss.get_cos_sim(array_dict['B'], array_dict['C']))
    ]

    result = fss.get_state_similarities(test_frame)
    assert result == expected


def test_get_sim_info():
    test = [
        ('A', 'B', 1),
        ('A', 'D', .5),
        ('A', 'C', .25)
    ]

    most_expected = [
        ('A', 'B'),
        ('A', 'D')
    ]

    least_expected = [
        ('A', 'C'),
        ('A', 'D')
    ]

    most, least = fss.get_sim_info(test, 2)
    assert most_expected == most
    assert least_expected == least

