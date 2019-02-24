import unittest
import numpy as np
import pandas as pd
import find_similar_states as fss

def test_filter_by_race():
    test_frame = pd.DataFrame({
        'White': [], 'Black': [], 'Hispanic': [], 'Asian': [], 
        'American_Indian_or_Alaska_Native': [], 
        'Native_Hawaiian_or_Other_Pacific_Islander': [], 
        'Two_Or_More_Races': [], 'male':[],
        'female': []
    })

    expected = pd.DataFrame({
        'White': [], 'Black': [], 'Hispanic': [], 'Asian': [], 
        'American_Indian_or_Alaska_Native': [], 
        'Native_Hawaiian_or_Other_Pacific_Islander': [], 
        'Two_Or_More_Races': [],
    })

    result = fss.filter_by_race(test_frame)
    pd.testing.assert_frame_equal(result.sort_index(axis=1), expected.sort_index(axis=1))

def test_cos_sim():
    array_1 = np.array([1, 0, 0])
    array_2 = np.array([1, 2, 2])

    expected = 1.0/3
    result = fss.get_cos_sim(array_1, array_2)
    assert result == expected

def test_get_state_similarities():
    test_frame = pd.DataFrame({
        'White': [1,1,10,4], 'Black': [1,1,20,4], 'Hispanic': [0,0,50,1]
    })
    indeces = ['A', 'B', 'C', 'D']
    test_frame.index = indeces
    
    array_dict = {}
    for idx in indeces:
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
        ('A', 'D', .25),
        ('A', 'C', .5)
    ]

    most_expected = [
        ('A', 'B'),
        ('A', 'C'),
        ('A', 'D')
    ]

    least_expected = [
        ('A', 'D'),
        ('A', 'C'),
        ('A', 'B'),
    ]

    most, least = fss.get_sim_info(test)
    assert most_expected == most
    assert least_expected == least

if __name__ == '__main__':
    test_filter_by_race()
    test_cos_sim()
    test_get_state_similarities()
    test_get_sim_info()

