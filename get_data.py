import sys

import numpy as np

import __init__

sys.path.append(__init__.config['data_path'])  # add your data path here
from datasets import as_dataset
import pandas as pd
from tqdm import tqdm

batch_size = 1024

train_data_param = {
    'gen_type': 'train',
    'random_sample': True,
    'batch_size': batch_size,
    'split_fields': False,
    'on_disk': True,
    'squeeze_output': True,
}
test_data_param = {
    'gen_type': 'test',
    'random_sample': False,
    'batch_size': batch_size,
    'split_fields': False,
    'on_disk': True,
    'squeeze_output': True,
}

dataset = as_dataset('criteo')

train_gen = dataset.batch_generator(train_data_param)
test_gen = dataset.batch_generator(test_data_param)

xs = []
ys = []
for x, y in tqdm(train_gen):
    xs.append(x)
    ys.append(y)

x = np.concatenate(xs, 0)
y = np.concatenate(ys, 0)
print(x.shape)
np.save('train_x.npy',x)
np.save('train_y.npy', y)


xs = []
ys = []
for x, y in tqdm(test_gen):
    xs.append(x)
    ys.append(y)

x = np.concatenate(xs, 0)
y = np.concatenate(ys, 0)
print(x.shape)
np.save('test_x.npy',x)
np.save('test_y.npy', y)

# for x, y in tqdm(train_gen):
#     df = pd.DataFrame(x)
#     df['label'] = y
#     df.to_csv('train.csv', mode='a', header=None, index=None)
#
#
# for x, y in tqdm(test_gen):
#     df = pd.DataFrame(x)
#     df['label'] = y
#     df.to_csv('test.csv', mode='a', header=None, index=None)