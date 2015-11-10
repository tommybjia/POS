from __future__ import division
import numpy as np

def get_count_table_from(data_path):
	word_pool, tag_pool = {}, {}
	num_of_lines = sum(1 for line in open(data_path))
	e_counts, t_counts, end_counts = np.zeros((num_of_lines,num_of_lines)), np.zeros((num_of_lines,num_of_lines)), np.zeros(num_of_lines)
	precedent_tag = 0
	for line in open(data_path,'r'):
		perline = line.strip().split(' ')
		if perline[0] != '':
			if perline[1] not in tag_pool:
				tag_pool[perline[1]] = len(tag_pool)
			if perline[0] not in word_pool:
				word_pool[perline[0]] = len(word_pool)
			e_counts[tag_pool[perline[1]]][word_pool[perline[0]]] += 1
			t_counts[tag_pool[perline[1]]][precedent_tag] += 1
			precedent_tag = tag_pool[perline[1]] + 1
		else:
			end_counts[precedent_tag] += 1 
			precedent_tag = 0
	e_counts = e_counts[:len(tag_pool),:len(word_pool)]
	e_counts[e_counts == 0] = 1
	t_counts = np.vstack([t_counts[:len(tag_pool),:len(tag_pool)+1],end_counts[:len(tag_pool)+1]])
	return e_counts, t_counts, word_pool, tag_pool

def learn_parameters_from(data_path):
	e_counts, t_counts, word_pool, tag_pool = get_count_table_from(data_path)
	#emission parameters
	emission_params = e_counts/np.repeat(np.sum(e_counts,axis=1)+1,len(word_pool)).reshape([len(tag_pool),len(word_pool)])
	#transition parameters
	transition_params = t_counts/np.tile(np.sum(t_counts,axis=0),[t_counts.shape[0],1])
	return emission_params, transition_params
	
def simple_POS_tagger(emission_params, word_pool, tag_pool):
	POS_tag_predictions = map(lambda a:(list(k for k, v in tag_pool.iteritems() if v == a)[0]), np.argmax(emission_params,axis=0))
	return dict(zip(sorted(word_pool, key=lambda key: word_pool[key]), POS_tag_predictions))

def viterbi_tagger_loops(word_sequence, emission, transition, word_pool, tag_pool):
	pi = np.zeros((len(word_sequence),len(tag_pool)))
	for l in range(1, len(word_sequence)):
		for k,tag in tag_pool.iteritems():
			if (l == 1):
				pi[l][tag] = transition[tag][0]*emission[tag][word_pool[word_sequence[l-1]]]
			else:
				for pre_k,pre_tag in tag_pool.iteritems():
					if pi[l-1][pre_tag]*transition[tag][pre_tag+1]*emission[tag][word_pool[word_sequence[l-1]]] > pi[l][tag]:
						pi[l][tag] = pi[l-1][pre_tag]*transition[tag][pre_tag+1]*emission[tag][word_pool[word_sequence[l-1]]]
				if l == len(word_sequence)-1:
					pi[l][tag] = pi[l][tag]*transition[len(tag_pool)][tag+1]
	return map(lambda a:list(k for k,v in tag_pool.iteritems() if v == a)[0],np.argmax(pi,axis=1))

#learned parameters
emission_params, transition_params = learn_parameters_from('/Users/linyijuan/desktop/POS_dataset/train')
#simple POS tagger
e_counts, t_counts, word_pool, tag_pool = get_count_table_from('/Users/linyijuan/desktop/POS_dataset/train')
pos_tagger = simple_POS_tagger(emission_params, word_pool, tag_pool)
#viterbi_algo
word_sequence = ['good','friday','whatchu','got','for','me','@kanyewest']
tag_sequence = viterbi_tagger_loops(word_sequence, emission_params, transition_params, word_pool, tag_pool)
#print tag_sequence
