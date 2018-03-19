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
    plt.hist(data, bins=100, normed=1, alpha=1.0)
    plt.xlabel('Lengths')
    plt.ylabel('Probability')
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
    data = [s[:, 0], s[:,1]]
    bins = np.linspace(0, 300, 50)
    label=['begin', 'end']
    plt.hist(data, label=label, bins=bins, normed=1, alpha=1.0)
    plt.xlabel('Lengths')
    plt.ylabel('Probability')
    plt.legend(prop={'size': 20})
    plt.grid(True)
    plt.savefig(save_path + 'spans.png')
    plt.clf()

    answers= []
    keywords = ["why", "when", "how", "what", "who", "where"]
    for key in keywords:
        answers.append([len(answer) for (i, answer) in enumerate(a) if key in q[i]])
    bins = np.linspace(0, 30, 15)
    plt.hist(answers, bins, normed=True, label=keywords)
    plt.xlabel('Lengths')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path + 'answers_questiontype.png')
    plt.clf()

    contexts= []
    keywords = ["why", "when", "how", "what", "who", "where"]
    for key in keywords:
        contexts.append([len(context) for (i, context) in enumerate(c) if key in q[i]])
    bins = np.linspace(20, 300, 15)
    plt.hist(contexts, bins, normed=True, label=keywords)
    plt.xlabel('Lengths')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path + 'contexts_questiontype.png')
    plt.clf()

    answers= []
    keywords = ["why", "when", "how", "what", "who", "where"]
    for key in keywords:
        answers.append([len(answer) for (i, answer) in enumerate(q) if key in q[i]])
    bins = np.linspace(20, 30, 15)
    plt.hist(answers, bins, normed=True, label=keywords)
    plt.xlabel('Lengths')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path + 'questions_questiontype.png')
    plt.clf()
