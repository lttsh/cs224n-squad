import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import find

end_probs = np.load('experiments/baseline/end_span.npy')
begin_probs = np.load('experiments/baseline/begin_span.npy')
f1_em = np.load('experiments/baseline/f1_em.npy')
c2q_dist = np.load('experiments/baseline/c2q_attn.npy')

end_answer = np.argmax(end_probs, axis=1)
begin_answer = np.argmax(begin_probs, axis=1)
model_answer = np.stack([begin_answer, end_answer], axis=1)
print model_answer[:10]
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
def load_data(mode = 'dev'):
    path = 'data/' + mode
    answers = (load_file(path + '.answer'))
    contexts = (load_file(path + '.context'))
    questions = (load_file(path + '.question'))
    spans = load_file(path + '.span')
    return answers, contexts, questions, spans

answers, contexts, questions, spans = load_data()

'''
Filter based on a condition function
'''

def get_samples_with_conditions(samples, condition, num_samples=0, random=True):
    indices = np.where(condition(samples))[0]
    if num_samples != 0:
        if random:
            np.random.choice(indices, num_samples)
        else:
            return indices[:num_samples]
    else:
        return indices

def get_question_type_stat(keyword):
    questions_ind = get_samples_with_conditions(questions, lambda s : find(s[:], keyword) >= 0)
    f1_em_type = [f1_em[i] for i in questions_ind]
    print np.mean(f1_em_type, axis=0)


def get_error_type():
    end_before = get_samples_with_conditions(model_answer, lambda s: s[:,0] > s[:,1])
    print model_answer[end_before[0]]
# exact_match_examples = get_samples_with_conditions(f1_em, lambda s : s[:, 0] == 1.0)

get_error_type()
get_question_type_stat("why ")
get_question_type_stat("what ")
get_question_type_stat("when ")
get_question_type_stat("how ")
get_question_type_stat("who ")
