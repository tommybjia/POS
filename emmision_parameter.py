import numpy as np

def get_all_words(filepaths):
	x = []
	for file in filepaths:
		f_open = open(file, 'r')
		total_line = sum(1 for line in open(file))
		for i in range(total_line):
			x.append(f_open.readline().strip())
	return x

def emmision_parameters(data_path, alphabet):
	f_open = open(data_path, 'r')
	total_line = sum(1 for line in open(data_path))
	y = {}		# {tag: {word: num}}
	for i in range(total_line):
		line = f_open.readline().strip().split(' ')
		if len(line) > 1:
			if line[1] not in y:
				y[line[1]] = {}
				y[line[1]][line[0]] = 1
			else:
				if line[0] not in y[line[1]]:
					y[line[1]][line[0]] = 1
				else:
					y[line[1]][line[0]] += 1
	
	#fix probability for all new words
	e = []		# [tag][word]
	for key in y:
#		print y[key]
		for sub_key in y[key]:
#			print y[key][sub_key]
			if (y[key][sub_key] in alphabet):
				e[key] = {sub_key:y[key][sub_key]}
				
			tmp = y[key][sub_key]/
			e.append()


def main():
	emmision_parameters('/Users/abc/Desktop/POS
	_dataset/train',get_all_words('/Users/abc/Desktop/POS_dataset/dev.in'))




if __name__ == '__main__':
	main()
