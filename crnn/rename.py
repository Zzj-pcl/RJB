#rename
import os, io

for root, dirs, files in os.walk('G:\why_workspace\RJB\iam\images'):
    for file in files:
        src = root + '/' + file
        dst = root + '/' + file[:6] + '.png'
        os.rename(src, dst)
        # print('src:', src)
        # print('dst:', dst)

#relabel
import os, io

train_data_dir = "G:/why_workspace/RJB/dataset2/images/"
count = 0
for root, dirs, files in os.walk(train_data_dir):
    for file in files:
        count += 1
        src = root + file
        dst = root + '%06d' % count + '.png'
        with open('G:/why_workspace/RJB/dataset2/label.txt', 'a') as f:
            f.write('%06d' % count + '.png ' + file[:4] + '\n')
        print(src)
        print(dst)
        os.rename(src, dst)

# label_len 4->3
import os, io
with open('G:/why_workspace/RJB/dataset2/label.txt', 'r') as f:
    data = f.readlines()
    for line in data:
        write = line[:11]
        print(line)
        str = line[11:].split("_")
        for _ in str:
            write += _
        print(write)
        with open('G:/why_workspace/RJB/dataset2/label2.txt', 'a') as f:
            f.write(write)