import pandas as pd
import numpy as np


def extract_subjects(paths_df):
    subjects_df = pd.DataFrame()

    for i, path in enumerate(paths_df.iloc[:,0]):
        components = path.split('/')
        '''
        Example path:
        /home/arman_avesta/capsnet/data/images/012_S_1009/2008-12-16_06_42_02.0/brainbox.nii.gz'
        '''
        subject = data=components[-3]
        subjects_df.at[i, 'Subject'] = subject

    assert len(paths_df) == len(subjects_df)
    return subjects_df


def select_subjects(dem, subjects):
    selected_dem = pd.DataFrame()
    subjects = list(subjects.iloc[:,0])
    for i, subject in enumerate(subjects):
        temp = dem[dem['Subject'] == subject]
        selected_dem = pd.concat([selected_dem, temp])
    return selected_dem


dem_path = '/data/datasets/archives/FS_brainmasks_Demographics.xlsx'
train_path = '/Users/sa936/projects/capsnet/data/datasets/train_inputs.csv'
valid_path = '/Users/sa936/projects/capsnet/data/datasets/valid_inputs.csv'
test_path = '/Users/sa936/projects/capsnet/data/datasets/test_inputs.csv'

dem = pd.read_excel(dem_path)
train = pd.read_csv(train_path, header=None)
valid = pd.read_csv(valid_path, header=None)
test = pd.read_csv(test_path, header=None)

train_subjects = extract_subjects(train).drop_duplicates()
valid_subjects = extract_subjects(valid).drop_duplicates()
test_subjects = extract_subjects(test).drop_duplicates()
'''
dropping duplicates because the demographics dataframe already has duplicate rows for patients who were scanned
more than once. If we don't drop duplicates, we would have duplicates of duplicates!
'''

train_dem = select_subjects(dem, train_subjects)
valid_dem = select_subjects(dem, valid_subjects)
test_dem = select_subjects(dem, test_subjects)

train_age_mean, valid_age_mean, test_age_mean = \
    train_dem.Age.mean(), valid_dem.Age.mean(), test_dem.Age.mean()
train_age_std, valid_age_std, test_age_std = \
    train_dem.Age.std(), valid_dem.Age.std(), test_dem.Age.std()

train_MF, valid_MF, test_MF = \
    train_dem.Sex.value_counts()/len(train_dem)*100, \
    valid_dem.Sex.value_counts() / len(valid_dem) * 100, \
    test_dem.Sex.value_counts()/len(test_dem)*100

train_dx, valid_dx, test_dx = \
    train_dem.Group.value_counts() / len(train_dem) * 100, \
    valid_dem.Group.value_counts() / len(valid_dem) * 100, \
    test_dem.Group.value_counts() / len(test_dem) * 100
