def emmision_parameters(data_path):
	f_open = open(data_path, 'r')
	total_line = sum(1 for line in open(data_path))
	y = {}		# {tag: {word: num}}
	e = []		# [tag][word]
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
	# e = []
	# for key in y:
	# 	# print y[key]
	# 	for sub_key in y[key]:
	# 		# print y[key][sub_key]
	# 		tmp = y[key][sub_key]/
	# 		e.append()





def main():
	emmision_parameters('/Users/abc/Desktop/POS/train')




if __name__ == '__main__':
	main()
