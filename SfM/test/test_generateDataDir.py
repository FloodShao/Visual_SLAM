import os
from config_default import file_dir

if __name__ == '__main__':

    associate_file = file_dir['data_dir_test']


    fh = open(associate_file, 'w')
    N = 5
    for i in range(N):
        data_dir = "./data/" + str(i+1) + ".png\n"
        fh.write(data_dir)
    fh.close()

    fh = open(associate_file, 'r')
    '''
    if fh:
        print(1)
    else:
        print(0)

    if fh.readlines():
        print(1)
    else:
        print(0)
    '''
    for line in fh.readlines():
        if line:
            line = line.strip()
            print(line)
        else:
            print("line", 0)


    fh.close()


    



