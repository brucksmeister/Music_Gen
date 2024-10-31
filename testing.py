from data_class import *

chords_index_folder = '/Users/andi/PycharmProjects/Data_gen_jambot/data/shifted/chord_index'

chord_train, chord_test = get_chord_train_and_test_set(1000,10)
chord_train


chords_folder_cassette = '/Users/andi/PycharmProjects/Data_gen_jambot/data/shifted/chords_cassette'
train, test = get_chord_train_and_test_set_cassette(10,1)
train