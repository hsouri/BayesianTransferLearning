''' Code from: 
https://jovian.ai/yohannesmelese4/project-from-scratch

Please copy this script in your desired directory and it will create train and val foulders in that directory.
'''
import os
import pandas as pd
from torchvision.datasets.utils import download_and_extract_archive
import shutil


URL = "https://s3.amazonaws.com/fast-ai-imageclas/oxford-102-flowers.tgz"
download_and_extract_archive(URL, '.')


#os.listdir('oxford-102-flowers/train.txt')
train_dir = pd.read_csv('oxford-102-flowers/train.txt', header=None, delimiter=' ')
valid_dir = pd.read_csv('oxford-102-flowers/valid.txt', header=None, delimiter=' ')
test_dir = pd.read_csv('oxford-102-flowers/test.txt', header=None, delimiter=' ')

train_dir.shape, valid_dir.shape, test_dir.shape

label = train_dir[1]
each_label = label.value_counts()
print(f'no of classes {len(each_label)}')
print(each_label)

def add_zero(n):
  return '0'*(3 - len(str(n))) + str(n)

# lets create a train, valid and test directories

data_pt = ['train', 'val', 'test']
for main_pt in data_pt:
    os.mkdir(main_pt)
    for lab in range(102):
      os.mkdir(main_pt+'/'+add_zero(lab))




main_path = 'oxford-102-flowers/'

train_path = 'train/'
valid_path = 'val/'
test_path = 'test/'

# lets move the images to the given directories

for img, lab in train_dir.values:
    shutil.move(main_path+img, train_path+add_zero(lab))

for x, (img, lab) in enumerate(valid_dir.values):
    shutil.move(main_path+img, valid_path+add_zero(lab))

for x, (img, lab) in enumerate(test_dir.values):
    shutil.move(main_path+img, test_path+add_zero(lab))