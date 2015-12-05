import numpy as np
import copy
import time

def train_pool(filepath):
    # word_pool_train and tag_pool_train are dictionaries for training data. The key is the word or tag and the value is the index of word and tag in the counting table.
    word_pool_train = {}
    tag_pool_train = {'START':0}
    num_of_lines = sum(1 for line in open(filepath))
    # emission_count[tag][word] = num of word in this tag
    # transition_count[tag(i)][tag(i+1)] = num of tag(i+1) after tag(i), which tag(0) is start and tag(n+1) is stop
    emission_count = np.zeros((num_of_lines+1, num_of_lines+1), dtype=float)
    transition_count = np.zeros((num_of_lines+1, num_of_lines+1), dtype=float)

    for line in open(filepath, 'r'):
        perline = line.strip().split(' ')
        if perline[0].lower() != '':
            if perline[1] not in tag_pool_train:
                tag_pool_train[perline[1]] = len(tag_pool_train)
            if perline[0].lower() not in word_pool_train:
                word_pool_train[perline[0].lower()] = len(word_pool_train)
            emission_count[tag_pool_train[perline[1]], word_pool_train[perline[0].lower()]] += 1
        
    tag_pool_train['STOP'] = len(tag_pool_train)
    emission_count = emission_count[:len(tag_pool_train), :len(word_pool_train)]

    prev_tag = ''
    now_tag = ''
    for line in open(filepath, 'r'):
        perline = line.strip().split(' ')
        if perline[0].lower() != '':
            now_tag = perline[1]
        else:
            now_tag = ''
        if prev_tag == '' and now_tag != '':
            transition_count[tag_pool_train['START'], tag_pool_train[now_tag]] += 1
        elif prev_tag != '' and now_tag == '':
            transition_count[tag_pool_train[prev_tag], tag_pool_train['STOP']] += 1
        elif prev_tag != '' and now_tag != '':
            transition_count[tag_pool_train[prev_tag], tag_pool_train[now_tag]] += 1
        prev_tag = now_tag

    transition_count = transition_count[:len(tag_pool_train), :len(tag_pool_train)]
    most_common_tag_index = np.argmax(np.sum(transition_count, axis=1))

    return word_pool_train, tag_pool_train, emission_count, transition_count, most_common_tag_index

def cal_params(train_filepath, test_filepath):
    word_pool_train, tag_pool_train, emission_count, transition_count, most_common_tag_index = train_pool(train_filepath)
    # create a new pool that contain the testing data word
    word_pool_test = copy.copy(word_pool_train)
    num_of_lines = sum(1 for line in open(test_filepath))
    emission_params = np.zeros((num_of_lines*10, num_of_lines*10), dtype=float)
    transition_params = copy.copy(transition_count)
    # compute emission parameter table
    for line in open(test_filepath, 'r'):
        perline = line.strip().split(' ')
        if perline[0].lower() != '':
            if perline[0].lower() not in word_pool_test:
                word_pool_test[perline[0].lower()] = len(word_pool_test)
                for i in range(1,len(tag_pool_train)-1):
                    emission_params[i, word_pool_test[perline[0].lower()]] = 1 / (1 + np.sum(emission_count[i]))
    for word in word_pool_train:
        for i in range(1, len(tag_pool_train)-1):
            emission_params[i, word_pool_train[word.lower()]] = emission_count[i, word_pool_train[word.lower()]] / (1 + np.sum(emission_count[i]))

    emission_params = emission_params[:len(tag_pool_train), :len(word_pool_test)]
    # compute transition parameters table
    for i in range(len(transition_params)-1):
        transition_params[i] = transition_params[i]/np.sum(transition_params[i])

    return emission_params, transition_params, word_pool_test, tag_pool_train, most_common_tag_index

def pos_tagger(train_filepath, test_filepath, accuracy_filepath):
    emission_params, transition_params, word_pool_test, tag_pool_train, most_common_tag_index = cal_params(train_filepath, test_filepath)
    zerozero = 0
    count = 0
    correct = float(0)
    wrong = float(0)
    file_out = open('./dev.p2.out','w')
    for line in open(test_filepath, 'r'):
        perline = line.strip().split(' ')
        if perline[0].lower() != '':
            pred_tag_index = np.argmax(emission_params[:, word_pool_test[perline[0].lower()]])
            for key, value in tag_pool_train.iteritems():
                if value == pred_tag_index:
                    pred_tag = key
                    break
            file_out.write(perline[0].lower() + ' ' + pred_tag + '\n')
        else:
            file_out.write('\n')
    file_out.close()

    tag_set = []
    tag_set_pred = []
    for line in open(accuracy_filepath,'r'):
        perline = line.strip().split(' ')
        if perline[0].lower() != '':
            tag_set.append(perline[1])
    for line in open('./dev.p2.out','r'):
        perline = line.strip().split(' ')
        if perline[0].lower() != '':
            tag_set_pred.append(perline[1])

    for i in range(len(tag_set)):
        if tag_set[i] == tag_set_pred[i]:
            correct += 1
        else:
            wrong +=1

    return correct/(correct + wrong)

def viterbi(word_sequence, emission_params, transition_params, word_pool_test, tag_pool_train, most_common_tag_index):
    PI = np.zeros((len(word_sequence) + 1, len(tag_pool_train) + 1))
    paths = np.zeros((len(word_sequence) + 1, len(tag_pool_train) + 1))
    paths[:,:] = most_common_tag_index

    for k in range(1, len(word_sequence)+1):
        for v in range(1, len(tag_pool_train) - 1):
            if k == 1:
                PI[k,v] = transition_params[0, v] * emission_params[v, word_pool_test[word_sequence[k-1].lower()]]
            else:
                max_PI = float(0)
                max_path_index = most_common_tag_index
                for u in range(1, len(tag_pool_train)-1):
                    tmp_PI = PI[k-1, u] * transition_params[u, v] * emission_params[v, word_pool_test[word_sequence[k-1].lower()]]
                    if tmp_PI > max_PI:
                        max_PI = tmp_PI
                        max_path_index = u
                PI[k, v] = max_PI
                paths[k, v] = max_path_index

    max_PI = float(0)
    max_path_index = most_common_tag_index
    for v in range(1, len(tag_pool_train)-1):
        tmp_PI = PI[len(word_sequence), v] * transition_params[v, len(tag_pool_train)-1]
        if tmp_PI > max_PI:
            max_PI = tmp_PI
            max_path_index = v

    best_path = [max_path_index]
    prev_node = max_path_index
    for i in range(0, len(word_sequence)-1):
        best_path.append(paths[len(word_sequence)-i, prev_node])
        prev_node = paths[len(word_sequence)-i, prev_node]
    best_path.reverse()

    best_path_tag = []
    for i in range(len(best_path)):
        for key, value in tag_pool_train.iteritems():
            if best_path[i] == value:
                best_path_tag.append(key)

    return best_path_tag


def viterbi_top_nth(word_sequence, emission_params, transition_params, word_pool_test, tag_pool_train, most_common_tag_index, num_of_opt_path=10, require_path_index=5):
    PI = np.zeros((len(word_sequence)+1, len(tag_pool_train)+1, num_of_opt_path))
    paths = np.zeros((len(word_sequence)+1, len(tag_pool_train)+1, num_of_opt_path, 2))
    paths[:,:,:,:] = most_common_tag_index

    for k in range(1, len(word_sequence)+1):
        for v in range(1, len(tag_pool_train)-1):
            if k == 1:
                PI[k,v] = transition_params[0,v] * emission_params[v, word_pool_test[word_sequence[k-1].lower()]]
            elif k == 2:
                for u in range(1, len(tag_pool_train)-1):
                    tmp_PI = PI[k-1, u, 0] * transition_params[u,v] * emission_params[v, word_pool_test[word_sequence[k-1].lower()]]
                    for m in range(num_of_opt_path):
                        if tmp_PI > PI[k,v,m]:
                            for d in range(0, num_of_opt_path-m-1):
                                PI[k,v,num_of_opt_path-d-1] = PI[k,v,num_of_opt_path-d-2]
                                paths[k,v,num_of_opt_path-d-1] = paths[k,v,num_of_opt_path-d-2]
                            PI[k,v,m] = tmp_PI
                            paths[k,v,m] = (u,0)
                            break
            else:
                for u in range(1, len(tag_pool_train)-1):
                    for n in range(num_of_opt_path):
                        tmp_PI = PI[k-1, u, n] * transition_params[u,v] * emission_params[v,word_pool_test[word_sequence[k-1].lower()]]
                        for m in range(num_of_opt_path):
                            if tmp_PI >= PI[k, v, m]:
                                for d in range(0, num_of_opt_path-m-1):
                                    PI[k, v, num_of_opt_path-d-1] = PI[k, v, num_of_opt_path-d-2]
                                    paths[k, v, num_of_opt_path-d-1] = paths[k, v, num_of_opt_path-d-2]
                                PI[k, v, m] = tmp_PI
                                paths[k, v, m] = (u, n)
                                break

    max_PI = np.zeros((num_of_opt_path))
    max_path_index = np.zeros((num_of_opt_path, 2))
    for v in range(1, len(tag_pool_train)-1):
        for n in range(num_of_opt_path):
            tmp_PI = PI[len(word_sequence), v, n] * transition_params[v, len(tag_pool_train)-1]
            for m in range(num_of_opt_path):
                if tmp_PI >= max_PI[m]:
                    for d in range(num_of_opt_path-m-1):
                        max_PI[num_of_opt_path-d-1] = max_PI[num_of_opt_path-d-2]
                        max_path_index[num_of_opt_path-d-1] = max_path_index[num_of_opt_path-d-2]
                    max_PI[m] = tmp_PI
                    max_path_index[m] = (v, n)
                    break

    best_path = [max_path_index[require_path_index-1,0]]
    prev_node = max_path_index[require_path_index-1, 0]
    prev_index = max_path_index[require_path_index-1, 1]

    for i in range(0, len(word_sequence)-1):
        best_path.append(paths[len(word_sequence)-i, prev_node, prev_index, 0])
        prev_node = paths[len(word_sequence)-i, prev_node, prev_index, 0]
        prev_index = paths[len(word_sequence)-i, prev_node, prev_index, 1]
    best_path.reverse()

    best_path_tag = []
    for i in range(len(best_path)):
        for key, value in tag_pool_train.iteritems():
            if best_path[i] == value:
                best_path_tag.append(key)

    return best_path_tag

def calAccuracyInViterbi(train_filepath, test_filepath, accuracy_filepath, viterbifunction=viterbi):

    emission_params, transition_params, word_pool_test, tag_pool_train, most_common_tag_index = cal_params(train_filepath, test_filepath)
    
    word_sequence = []
    word_sequence_pool = {}
    countcount = 1
    word_sequence = []
    for line in open(test_filepath, 'r'):
        perline = line.strip().split(' ')
        if perline[0] != '':
            word_sequence.append(perline[0])
        else:
            word_sequence_pool[countcount] = word_sequence
            countcount += 1
            word_sequence = []


    # compute the accuracy
    file_out = open('./dev.p3.out','w')
    file_comp = open(accuracy_filepath,'r')
    correct = 0
    wrong = 0
    count = 0
    for sentence in word_sequence_pool.values():
        best_path_tag = viterbifunction(sentence, emission_params, transition_params, word_pool_test, tag_pool_train, most_common_tag_index)
        count += 1
        print 'No. %d sentence.' %count
        for i in range(len(sentence)):
            tag = file_comp.readline().strip().split(' ')
            file_out.write(sentence[i] + ' ' + best_path_tag[i] + '\n')
            if tag[1] == best_path_tag[i]:
                correct += 1
            else:
                wrong += 1
        file_out.write('\n')
        file_comp.readline()

    # for line in open(test_filepath,'r'):
    #     perline = line.strip().split(' ')
    #     if perline[0] != '':
    #         word_sequence.append(perline[0])
    #     else:
    #         best_path_tag = viterbifunction(word_sequence, emission_params, transition_params, word_pool_test, tag_pool_train, most_common_tag_index)
    #         count += 1
    #         print 'No. %d sentence.' %count
    #         for i in range(len(word_sequence)):
    #             tag = file_comp.readline().strip().split(' ')
    #             file_out.write(word_sequence[i] + ' ' + best_path_tag[i] + '\n')
    #             if tag[1] == best_path_tag[i]:
    #                 correct += 1
    #             else:
    #                 wrong += 1
    #         file_out.write('\n')
    #         file_comp.readline()
    #         word_sequence = []

    return (float(correct)/(correct+wrong))

if __name__ == '__main__':
    POS_trainingpath = './POS_dataset/train'
    POS_testingpath = './POS_dataset/dev.in'
    POS_accuracypath = './POS_dataset/dev.out'

    NPC_trainingpath = './NPC_dataset/train'
    NPC_testingpath = './NPC_dataset/dev.in'
    NPC_accuracypath = './NPC_dataset/dev.out'

    # Testing POS tagger
    # print 'The accuracy of pos_tagger for dataset dev.in in POS is: ', pos_tagger(POS_trainingpath, POS_testingpath, POS_accuracypath)
    # print 'The accuracy of pos_tagger for dataset dev.in in NPC is: ', pos_tagger(NPC_trainingpath, NPC_testingpath, NPC_accuracypath)

    # Testing Viterbi
    # print 'The accuracy of viterbi for dataset dev.in in POS is: ', calAccuracyInViterbi(POS_trainingpath, POS_testingpath, POS_accuracypath)
    # print 'The accuracy of viterbi for dataset dev.in in NPC is: ', calAccuracyInViterbi(NPC_trainingpath, NPC_testingpath, NPC_accuracypath)

    # Testing Viterbi for 10th sequence
    print 'The accuracy of 10-th viterbi for dataset dev.in in POS is: ', calAccuracyInViterbi(POS_trainingpath, POS_testingpath, POS_accuracypath, viterbi_top_nth)
