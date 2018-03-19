import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from numpy.core.defchararray import find

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


'''
Filter based on a condition function
'''

def get_samples_with_conditions(samples, condition, num_samples=0, random=True):
    indices = np.where(condition(samples))[0]
    if num_samples > 0:
        if random:
            return np.random.choice(indices, num_samples)
        else:
            return indices[:num_samples]
    else:
        return indices

def get_question_type_stat(keyword, questions, f1_em):
    questions_ind = get_samples_with_conditions(questions, lambda s : find(s[:], keyword) >= 0)
    f1_em_type = [f1_em[i] for i in questions_ind]
    print np.mean(f1_em_type, axis=0)

def get_answer_length_stat(length, lengths, f1_em):
    questions_ind = [i for i in range(len(lengths)) if lengths[i] > length]
    f1_em_type = [f1_em[i] for i in questions_ind]
    print np.mean(f1_em_type, axis=0)

def get_error_type():
    end_before = get_samples_with_conditions(model_answer, lambda s: s[:,0] > s[:,1])
    print model_answer[end_before[0]]
# exact_match_examples = get_samples_with_conditions(f1_em, lambda s : s[:, 0] == 1.0)

def visualize_attention(attention, contexts, questions, index, filename):
    # Assume attention is of size (N, Context_len, Question_len)

    plt.clf()
    context = contexts[index]
    question = questions[index]
    if len(contexts[index]) > 300:
        context = contexts[index][:300]
    if len(question) > 20:
        question = questions[index][:20]
    real_context_len = len(context)
    real_question_len = len(question)
    df_cm = pd.DataFrame(attention[index, :real_context_len, :real_question_len], index = [i.decode("utf-8") for i in context],
                  columns = [i.decode("utf-8") for i in question])
    sn.set(font_scale=0.75)
    fig, ax = plt.subplots(
        figsize=(8,15))
    sn.heatmap(df_cm, annot=False, square=False, xticklabels=1, yticklabels=1, ax=ax,cmap="YlGnBu")
    plt.savefig(filename)
    plt.close()

def visualize_q2c_attention(attention, contexts, index, filename):
    # Assume attention is of size (N, Context_len)
    plt.clf()
    context = contexts[index]
    if len(contexts[index]) > 300:
        context = contexts[index][:300]
    real_context_len = len(context)
    df_cm = pd.DataFrame(attention[index, :real_context_len],
                  index = [i.decode("utf-8") for i in context])
    sn.set(font_scale=0.75)
    fig, ax = plt.subplots(
        figsize=(8,25))
    sn.heatmap(df_cm, annot=False, square=False, xticklabels=1, yticklabels=1, ax=ax,cmap="YlGnBu")
    plt.savefig(filename)
    plt.close()

def visualize_spans(begin_probs, end_probs, contexts, index, filename):
    plt.clf()
    context = contexts[index]
    if len(contexts[index])> 300:
        context = contexts[index][:300]
    real_context_len = len(context)
    print real_context_len
    probabilities = np.stack([begin_probs[index], end_probs[index]], axis=1)
    df_begin = pd.DataFrame(probabilities[:real_context_len].transpose(), index = ['begin', 'end'], columns= [i.decode("utf-8") for i in context])
    sn.set(font_scale=0.8)
    fig, ax = plt.subplots(
        figsize=(18,5))
    sn.heatmap(df_begin, annot=False, xticklabels=1, yticklabels=1, linewidths=0.01, ax=ax, cmap="YlGnBu")
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    answers, contexts, questions, spans = load_data()
    lengths = split_token(spans)
    lengths = [ int(l[1]) - int(l[0]) for l in lengths]
    print lengths[0]
    experiment_name = 'stack_more_size_less_context'
    end_probs = np.load('experiments/' + experiment_name + '/end_span.npy')
    begin_probs = np.load('experiments/' + experiment_name + '/begin_span.npy')
    f1_em = np.load('experiments/' + experiment_name + '/f1_em.npy')
    c2q_dist = np.load('experiments/' + experiment_name + '/c2q_attn.npy')
    print c2q_dist.shape
    q2c_dist = np.load('experiments/' + experiment_name + '/q2c_attn.npy')
    end_answer = np.argmax(end_probs, axis=1)
    begin_answer = np.argmax(begin_probs, axis=1)
    model_answer = np.stack([begin_answer, end_answer], axis=1)
    print model_answer[:10]

    get_error_type()
    print ("Overall stat")
    print np.mean(f1_em, axis=0)
    print ("Why stat")
    get_question_type_stat("why ", questions, f1_em)
    print ("What stat")
    get_question_type_stat("what ", questions, f1_em)
    print ("When stat")
    get_question_type_stat("when ", questions, f1_em)
    print ("How stat")
    get_question_type_stat("how ", questions, f1_em)
    print ("Who stat")
    get_question_type_stat("who ", questions, f1_em)
    print ("Where stat")
    get_question_type_stat("where ", questions, f1_em)

    for length in [1, 5, 10, 20]:
        print ("Stat for answers of length bigger than %d" % length)
        answer_lengths = [i for i in range(len(lengths)) if lengths[i] > length]
        print (np.mean(f1_em[answer_lengths], axis=0))

    success = get_samples_with_conditions(f1_em, lambda s: s[:, 0]<=0.7, num_samples =0, random=False)
    tokenized_context = split_token(contexts)
    short_contexts = [i for i in range(len(tokenized_context)) if  len(tokenized_context[i]) < 100]
    print ("There are %d short contexts" % len(short_contexts))

    question_word =""
    # why_questions = get_samples_with_conditions(questions, lambda s : find(s[:], question_word) >= 0, num_samples=0, random=False)
    indices = list(set(success).intersection(short_contexts))
    print indices
    save_path = 'experiments/' + experiment_name + '/visualization/'
    keyword='borderline_'+question_word
    for index in indices[:10]:
        print index
        visualize_attention(c2q_dist, split_token(contexts), split_token(questions), index, save_path+'attention' + keyword + str(index) + '.png')
        visualize_spans(begin_probs, end_probs, split_token(contexts), index, save_path + 'spans' + keyword + str(index) + '.png')
        visualize_q2c_attention(q2c_dist, split_token(contexts), index, save_path + 'q2c_attention' + keyword + str(index) + '.png')
