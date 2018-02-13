import matplotlib.pyplot as plt
import numpy as np
import time

'''
Utility functions to collect data on training set
'''

def load_file(path):
    return open(path, "r").read().split("\n")

def split_token(data, toint=False):
    data = data[:-1] #Last line is empty line
    return np.array([map(int, d.split(' ')) if toint else d.split(' ') for d in data])

def load_data(mode = 'train'):
    path = './data/' + mode
    answers = split_token(load_file(path + '.answer'))
    contexts = split_token(load_file(path + '.context'))
    questions = split_token(load_file(path + '.question'))
    spans = split_token(load_file(path + '.span'), toint=True)
    return answers, contexts, questions, spans

def plot_histogram(data, label=''):
    plt.hist(data, bins=100, normed=1, alpha=0.5)
    plt.xlabel('Lengths')
    plt.ylabel('Probability')
    plt.title('Histogram for ' + label)
    plt.grid(True)

start_time = time.time()
a, c, q, s = load_data()
print("Loaded data in %s seconds." % (time.time() - start_time))
plot_histogram(s[:, 0], 'begin-spans')
plot_histogram(s[:, 1], 'end-spans')
plt.show()
