#import numpy as np
import _pickle as pickle
import os

# settings

shifted = True
working_dir = os.getcwd()

shift_folder = ''
if shifted:
    shift_folder = 'shifted/'

# If you only want to process a subfolder like '/A' or '/A/A' for tests
subfolder = '/'

source_folder = os.path.join(working_dir, 'JamBot_experiments', 'data', 'original') + subfolder
tempo_folder1 = os.path.join(working_dir, 'JamBot_experiments', 'data/', 'tempo') + subfolder
histo_folder1 = os.path.join(working_dir, 'JamBot_experiments', 'data/', 'histo') + subfolder

tempo_folder2 = os.path.join(working_dir, 'JamBot_experiments', 'data', shift_folder, 'tempo') + subfolder
shifted_folder = os.path.join(working_dir, 'JamBot_experiments', 'data', shift_folder, 'shifted') + subfolder
pickle_folder = os.path.join(working_dir, 'JamBot_experiments', 'data', shift_folder, 'pianoroll') + subfolder
roll_folder = os.path.join(working_dir, 'JamBot_experiments', 'data', shift_folder, 'indroll') + subfolder
histo_folder2 = os.path.join(working_dir, 'JamBot_experiments', 'data', shift_folder, 'histo') + subfolder
chords_folder = os.path.join(working_dir, 'JamBot_experiments', 'data', shift_folder, 'chords') + subfolder
chords_index_folder = os.path.join(working_dir, 'JamBot_experiments', 'data', shift_folder, 'chord_index') + subfolder
song_histo_folder = os.path.join(working_dir, 'JamBot_experiments', 'data', shift_folder, 'song_histo') + subfolder

source_folder

######################################################### added #########################################################
chords_folder_cassette = os.path.join(working_dir, 'JamBot_experiments', 'data', shift_folder, 'chords_cassette') + subfolder
tempo_histo_folder = os.path.join(working_dir, 'JamBot_experiments', 'data', 'tempo_histo') + subfolder
########################################################################################################################


dict_path = os.path.join(working_dir, 'JamBot_experiments', 'data/')
chord_dict_name = 'chord_dict.pickle'
index_dict_name = 'index_dict.pickle'

if shifted:
    chord_dict_name = 'chord_dict_shifted.pickle'
    index_dict_name = 'index_dict_shifted.pickle'

# Adds the count of the beat as a feature to the input vector
counter_feature = True
counter_size = 0
if counter_feature:
    counter_size = 3

# Appends also the next cord to the feature vector:
next_chord_feature = True

high_crop = 84  # 84
low_crop = 24  # 24
num_notes = 128
new_num_notes = high_crop - low_crop
chord_embedding_dim = 10

# double_sample_chords = False
double_sample_notes = True

sample_factor = 2

one_hot_input = False
collapse_octaves = True
discretize_time = False
offset_time = False
discritezition = 8
offset = 16

# Some parameters to extract the pianorolls
# fs = 4 for 8th notes
fs = 4
samples_per_bar = fs * 2
octave = 12
melody_fs = 4

# Number of notes in extracted chords
chord_n = 3
# Number of notes in a key
key_n = 7
# Chord Vocabulary size
num_chords = 100

if shifted:
    num_chords = 50


cassette_chord_templates = [
    [1.0001, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 1.0001, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 1.0001, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 1.0001, 0, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 1.0001, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 1.0001, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 1.0001, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 1.0001, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 1.0001, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 1.0001, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1.0001, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1.0001],
    [1.0001, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 1.0001, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1.0001, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1.0001, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1.0001, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1.0001, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1.0001, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1.0001, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 1.0001, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1.0001, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1.0001, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1.0001]
]


def get_chord_train_and_test_set(train_set_size, test_set_size):
    data = make_chord_data_set()
    train_set = data[:train_set_size]
    test_set = data[train_set_size:train_set_size + test_set_size]
    return train_set, test_set


def make_chord_data_set():
    data = []
    for path, subdirs, files in os.walk(chords_index_folder):
        for name in files:
            if name.endswith('.pickle'):
                _path = path.replace('\\', '/') + '/'
                _name = name.replace('\\', '/')
                song = pickle.load(open(_path + _name, 'rb'))
                data.append(song)
    return data


def get_ind_train_and_test_set(train_set_size, test_set_size):
    data, chord_data = make_ind_data_set()
    train_set = data[:train_set_size]
    test_set = data[train_set_size:train_set_size + test_set_size]
    chord_train_set = chord_data[:train_set_size]
    chord_test_set = chord_data[train_set_size:train_set_size + test_set_size]
    return train_set, test_set, chord_train_set, chord_test_set


def make_ind_data_set():
    data = []
    chord_data = []
    for path, subdirs, files in os.walk(roll_folder):
        for name in files:
            if name.endswith('.pickle'):
                _path = path.replace('\\', '/') + '/'
                _name = name.replace('\\', '/')
                song = pickle.load(open(_path + _name, 'rb'))
                _chord_path = _path.replace('indroll', 'chord_index')
                song_chords = pickle.load(open(_chord_path + _name, 'rb'))
                data.append(song)
                chord_data.append(song_chords)
    return data, chord_data




#######added#####

def get_chord_train_and_test_set_cassette(train_set_size, test_set_size):
    data = make_chord_data_set_cassette()
    train_set = data[:train_set_size]
    test_set = data[train_set_size:train_set_size + test_set_size]
    return train_set, test_set


def make_chord_data_set_cassette():
    data = []
    for path, subdirs, files in os.walk(chords_folder_cassette):
        for name in files:
            if name.endswith('.pickle'):
                print(name)
                _path = path.replace('\\', '/') + '/'
                _name = name.replace('\\', '/')
                song = pickle.load(open(_path + _name, 'rb'))
                data.append(song)
    return data


def make_ind_data_set_cassette():
    data = []
    chord_data = []
    for path, subdirs, files in os.walk(roll_folder):
        for name in files:
            if name.endswith('.pickle'):
                _path = path.replace('\\', '/') + '/'
                _name = name.replace('\\', '/')
                song = pickle.load(open(_path + _name, 'rb'))
                _chord_path = _path.replace('indroll', 'chords_cassette')
                song_chords = pickle.load(open(_chord_path + _name, 'rb'))
                data.append(song)
                chord_data.append(song_chords)
    return data, chord_data
