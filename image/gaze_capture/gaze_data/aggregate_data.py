import h5py
import numpy as np
import os

collection = ''
gaze_data = []
for data_file in os.listdir():
    if 'h5' in data_file and ('1617' in data_file or 'large' in data_file):
    # if 'h5' in data_file and ('16166' in data_file):
        with h5py.File(data_file,'r') as data:
            data_dict = {int(k):v[()] for k,v in data.items()}
            gaze_data.append(data_dict)

data_np = {k[0]: np.concatenate(v) for k,v in [list(zip(*l)) for l in zip(*[list(d.items()) for d in gaze_data])]}
print(sum([d.shape[0] for d in data_np.values()]))
with h5py.File("Bottle_gaze_data_combine.h5",'w') as f:
    for k,v in data_np.items():
        f.create_dataset(str(k),data=v)
