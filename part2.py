from __future__ import division
import numpy as np

def learn_parameters_and_tagger_from(data_path):
	word_pool, tag_pool = {}, {}
	num_of_lines = sum(1 for line in open(data_path))
	counts = np.zeros((num_of_lines,num_of_lines))
	#store count of words and tags associatively into table
	for line in open(data_path,'r'):
		perline = line.strip().split(' ')
		if perline[0] != '':
			if perline[1] not in tag_pool:
				tag_pool[perline[1]] = len(tag_pool)
			if perline[0] not in word_pool:
				word_pool[perline[0]] = len(word_pool)
			counts[tag_pool[perline[1]]][word_pool[perline[0]]] += 1
	counts = counts[:len(tag_pool),:len(word_pool)]
	counts[counts == 0] = 1		
	#emmission parameters
	emmission_params = counts/np.repeat(np.sum(counts,axis=1)+1,len(word_pool)).reshape([len(tag_pool),len(word_pool)])
	#simple POS tagger
	POS_tag_predictions = np.concatenate([map(lambda a:(list(k for k, v in tag_pool.iteritems() if v == a)), np.argmax(emmission_params,axis=0))],axis=None) 
	simple_POS_tagger = dict(zip(sorted(word_pool, key=lambda key: word_pool[key]), POS_tag_predictions))
	
	return emmission_params,simple_POS_tagger
				
emmission_params, simple_POS_tagger = learn_parameters_from('/Users/linyijuan/desktop/POS_dataset/train')


