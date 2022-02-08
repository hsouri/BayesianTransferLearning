import os
import pandas as pd
from torchvision.datasets.utils import download_and_extract_archive
import shutil
import pdb


URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
download_and_extract_archive(URL, '.')
URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
download_and_extract_archive(URL, '.')


train_dir = pd.read_csv('annotations/trainval.txt', header=None, delimiter=' ')
test_dir = pd.read_csv('annotations/test.txt', header=None, delimiter=' ')



train_dir.shape, test_dir.shape

label = train_dir[1]
each_label = label.value_counts()
print(f'no of classes {len(each_label)}')
print(each_label)

def add_zero(n):
    return '0'*(2 - len(str(n))) + str(n)

# lets create a train, val directories

data_pt = ['train', 'val']
for main_pt in data_pt:
    os.mkdir(main_pt)
    for lab in range(37):
      os.mkdir(main_pt+'/'+add_zero(lab))




main_path = 'images/'

train_path = 'train/'
valid_path = 'val/'

# lets move the images to the given directories

for value in train_dir.values:
    shutil.move(main_path+value[0]+'.jpg', train_path+add_zero(value[1] - 1))

# we use test data as val images
for value in test_dir.values:
    shutil.move(main_path+value[0] +'.jpg', valid_path+add_zero(value[1] - 1))


