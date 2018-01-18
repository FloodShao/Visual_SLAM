file_dir = './data/rgb.txt'
out_dir = './data/rgb_track.txt'

fh = open(file_dir, 'r')
fo = open(out_dir, 'w')

for line in fh.readlines():
    a = './data/' + line[18:]
    fo.write(a)
    print(a)

fh.close()
fo.close()

