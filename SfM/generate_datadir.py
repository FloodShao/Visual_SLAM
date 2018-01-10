from config_default import file_dir

if __name__ == '__main__':

    associate_file = file_dir['data_dir_tube']

    fh = open(associate_file, 'w')
    N = 287
    for i in range(N):
        data_dir = "./data/capture_images_" + str(i) + ".jpg\n"
        fh.write(data_dir)

    fh.close()

