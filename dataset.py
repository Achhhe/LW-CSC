import torch.utils.data as data
import torch
import h5py
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get('data')
        self.target = hf.get('label')

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.target[index,:,:,:]).float()
        
    def __len__(self):
        return self.data.shape[0]

class DatasetFromPth(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromPth, self).__init__()

        result = torch.load(file_path)

        self.data = result['input']
        self.target = result['label']

    def __getitem__(self, index):
        return self.data[index,:,:,:].float(), self.target[index,:,:,:].float()
        
    def __len__(self):
        return self.data.numpy().shape[0]


class DatasetFromResult(data.Dataset):
    def __init__(self):
        super(DatasetFromResult, self).__init__()

    def __getitem__(self, index):
        return results[index].float(), results[index].float()
        
    def __len__(self):
        return len(results)

