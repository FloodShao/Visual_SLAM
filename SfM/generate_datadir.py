from config_default import file_dir

if __name__ == '__main__':

    associate_file = file_dir['data_dir_tube']

    fh = open(associate_file, 'w')
    N = 287
    for i in range(0, N):
        data_dir = "./data/capture_images_" + str(i) + ".jpg\n"
        fh.write(data_dir)

    fh.close()

    origin_file = './data/rgb.txt'
    associate_file = file_dir['data_dir']
    fi = open(origin_file, 'r')
    fo = open(associate_file, 'w')
    for line in fi.readlines():
        data_dir = "./data/" + line[18:]
        fo.write(data_dir)

    fi.close()
    fo.close()



