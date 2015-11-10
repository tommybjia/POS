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
    pool = {}       # {tag: {word: num}}
    x = [None]      # x[0]= None, x is the list of words
    y = [None]      # y[0] = None, y is the list of tags
    x_unique = []
    y_unique = []
    for i in range(total_line):
        line = f_open.readline().strip().split(' ')
        if line[0] != '':
            x.append(line[0])
            if line[0] not in x_unique:
                x_unique.append(line[0])
            y.append(line[1])
            if line[1] not in y_unique:
                y_unique.append(line[1])
        if len(line) > 1:
            if line[1] not in pool:
                pool[line[1]] = {}
                pool[line[1]][line[0]] = 1
            else:
                if line[0] not in pool[line[1]]:
                    pool[line[1]][line[0]] = 1
                else:
                    pool[line[1]][line[0]] += 1
    f_open.close()

    f_open = open(data_path, 'r')
    e_num = np.zeros((len(y_unique), len(x_unique)))    # e_num is the number count of e

    for i in range(total_line):
        line = f_open.readline().strip().split(' ')
        if line[0] != '':
            e_num[y_unique.index(line[1])][x_unique.index(line[0])] += 1

    e = np.zeros((len(y_unique), len(x_unique)))        # e is the parameter(e_num[i]/sum(e_num[i]))
    for i in range(len(x_unique)):
        e[:,i] = e_num[:,i] / np.sum(e_num, axis=1)


    


def main():
    file_path = '/Users/abc/Desktop/POS_dataset/train'
    # emmision_parameters('/Users/abc/Desktop/POS_dataset/train',get_all_words('/Users/abc/Desktop/POS_dataset/dev.in'))
    emmision_parameters(file_path)


if __name__ == '__main__':
    main()
