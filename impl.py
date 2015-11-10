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
		

def learn_parameters_and_tagger_from(data_path):
	e_counts, t_counts, word_pool, tag_pool = get_count_table_from(data_path)
	#emmission parameters
	emmission_params = e_counts/np.repeat(np.sum(e_counts,axis=1)+1,len(word_pool)).reshape([len(tag_pool),len(word_pool)])
	#transition parameters
	transition_params = t_counts/np.tile(np.sum(t_counts,axis=0),[t_counts.shape[0],1])
	#simple POS tagger
	POS_tag_predictions = np.concatenate([map(lambda a:(list(k for k, v in tag_pool.iteritems() if v == a)), np.argmax(emmission_params,axis=0))],axis=None) 
	simple_POS_tagger = dict(zip(sorted(word_pool, key=lambda key: word_pool[key]), POS_tag_predictions))
#	print simple_POS_tagger
	return emmission_params, transition_params, simple_POS_tagger
				
emmission_params, transition_params, simple_POS_tagger = learn_parameters_and_tagger_from('/Users/linyijuan/desktop/POS_dataset/train')

