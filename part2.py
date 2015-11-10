#!/usr/bin/python
import numpy as np
def parse_train_set(file):
	
	tags = []
	words = []
	
	lines = file.readlines()
	
#	construct tags and words list
	for line in lines:
		splited_line = line.split()
		if len(splited_line)==2:
			word,tag = splited_line[0].lower(),splited_line[1]
			if word not in words:
				words.append(word)
			if tag not in tags:
				tags.append(tag)
#		else:
#			print splited_line

#	construct tag-word matrix
	mat = np.zeros(shape=(len(tags),len(words)))
#	print mat.shape
	
#	scan the train again, put numbers into the matrix
	for line in lines:
		splited_line = line.split()
		if len(splited_line)==2:
			splited_line = line.split()
			word,tag = splited_line[0].lower(),splited_line[1]
			word_idx = words.index(word)
			tag_idx = tags.index(tag)
			mat[tag_idx,word_idx] = mat[tag_idx,word_idx] + 1
#			print word,tag,word_idx,tag_idx,mat[tag_idx,word_idx]
		
	return tags,words,mat
	
def calc_emit_param(tags,words,mat,x,y):
#	y is the test tag, x is the test word
#	this returns e(x|y) = count(y->x)/count(y)
	x = x.lower()
#	lower case of word
	if x in words:
		word_idx = words.index(x)
		tag_idx = tags.index(y)
		count_y_emit_x = mat[tag_idx,word_idx]
		count_y = mat[tag_idx,:].sum()
		
		emit_param = float(count_y_emit_x)/(count_y+1)
	
	else:
#		new word, not found in the train
		tag_idx = tags.index(y)
		
		count_y = mat[tag_idx,:].sum()
		emit_param = 1.0/(count_y+1)
	
#	
#	print count_y_emit_x
#	print count_y
	
	return emit_param

def pos_tagger(tags,words,mat,sentence):
#	sentense is a list of words
#	returns a list of tags
	rtn_tags = []
	
	for word in sentence:
		max_prob = 0
		max_tag = ""
		for tag in tags:
			prob = calc_emit_param(tags, words, mat, word, tag)
			if prob>max_prob:
				max_prob = prob
				max_tag = tag
		rtn_tags.append(max_tag)
	
	return rtn_tags

def calc_accuracy(tags,words,mat,gold_std_file):
	
	correct_count = 0
	prediction_count = 0
	
	lines = gold_std_file.readlines()
	for line in lines:
		splited_line = line.split()
		if len(splited_line)==2:
			test_word, gold_tag = splited_line[0].lower(),splited_line[1]
			predicted_tag = pos_tagger(tags, words, mat, [test_word])[0]
			if predicted_tag==gold_tag:
				prediction_count = prediction_count + 1
				correct_count = correct_count + 1
			else:
				prediction_count = prediction_count + 1
	
	accuracy = float(correct_count)/prediction_count
	return accuracy
	

training_set = file("train")
tags,words,mat = parse_train_set(training_set)
print calc_emit_param(tags, words, mat, "hello" , "NN")
print pos_tagger(tags, words, mat, ["I","eat","rice"])

gold_std = file("dev.out")
print "accuracy: ",calc_accuracy(tags, words, mat, gold_std)
