import numpy as np

def get_all_words(filepaths):
	x = []
	for file in filepaths:
		f_open = open(file, 'r')
		total_line = sum(1 for line in open(file))
		for i in range(total_line):
			x.append(f_open.readline().strip())
	return x

# def emmision_parameters(data_path, alphabet):
def emmision_parameters(data_path):
	f_open = open(data_path, 'r')
	total_line = sum(1 for line in open(data_path))
	pool = {}		# {tag: {word: num}}
	x = [None]		# x[0]= None, x is the list of words
	y = [None] 		# y[0] = None, y is the list of tags
	for i in range(total_line):
		line = f_open.readline().strip().split(' ')
		if line[0] != '':
			x.append(line[0])
			y.append(line[1])
		if len(line) > 1:
			if line[1] not in pool:
				pool[line[1]] = {}
				pool[line[1]][line[0]] = 1
			else:
				if line[0] not in pool[line[1]]:
					pool[line[1]][line[0]] = 1
				else:
					pool[line[1]][line[0]] += 1
	print x[0]
	


def main():
	file_path = '/Users/abc/Desktop/POS_dataset/train'
	# emmision_parameters('/Users/abc/Desktop/POS_dataset/train',get_all_words('/Users/abc/Desktop/POS_dataset/dev.in'))
	emmision_parameters(file_path)





if __name__ == '__main__':
	main()
