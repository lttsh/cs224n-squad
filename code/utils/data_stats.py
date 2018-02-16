import matplotlib.pyplot as plt
import numpy as np
import time

'''
Utility functions to collect data on training set
'''

def load_file(path):
    return open(path, "r").read().split("\n")

'''
Split sentences contained in data by token ' '
'''
def split_token(data, toint=False):
    data = data[:-1] #Last line is empty line
    return np.array([map(int, d.split(' ')) if toint else d.split(' ') for d in data])


'''
Load data in numpy array and splits them by token
'''
def load_data(mode = 'train'):
    path = '../data/' + mode
    answers = split_token(load_file(path + '.answer'))
    contexts = split_token(load_file(path + '.context'))
    questions = split_token(load_file(path + '.question'))
    spans = split_token(load_file(path + '.span'), toint=True)
    return answers, contexts, questions, spans


'''
Plots histogram for given data
'''
def plot_histogram(data, label=''):
    plt.hist(data, bins=100, normed=1, alpha=0.5)
    plt.xlabel('Lengths')
    plt.ylabel('Probability')
    plt.title('Histogram for ' + label)
    plt.grid(True)


if __name__ == "__main__":
    import os
    start_time = time.time()
    a, c, q, s = load_data()
    print("Loaded data in %s seconds." % (time.time() - start_time))
    save_path='stats/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plot_histogram([len(answer) for answer in a], 'answers')
    plt.savefig(save_path + 'answers.png')
    plt.clf()
    plot_histogram([len(context) for context in c], 'contexts')
    plt.savefig(save_path + 'contexts.png')
    plt.clf()
    plot_histogram([len(question) for question in q], 'questions')
    plt.savefig(save_path + 'questions.png')
    plt.clf()
    plot_histogram(s[:, 0], 'begin-spans')
    plt.savefig(save_path + 'begin_spans.png')
    plt.clf()
    plot_histogram(s[:, 1], 'end-spans')
    plt.savefig(save_path + 'end_spans.png')
    plt.clf()
