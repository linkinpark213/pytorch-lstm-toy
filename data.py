import numpy as np
import torch.utils.data


class SurnameDataset(torch.utils.data.Dataset):
    def __init__(self, subset='train'):
        self.data_path = 'data/{}/'.format(subset)
        self.tag_dict = {
            'English': 0,
            'Chinese': 1,
            'Japanese': 2
        }
        self.raw_data = []
        self.tags = []
        for key in self.tag_dict.keys():
            tag = self.tag_dict[key]
            surnames = self.read_file(self.data_path + key + '.txt')
            for surname in surnames:
                self.raw_data.append(surname)
                self.tags.append(tag)

    def read_file(self, file_path):
        """
        Read the surname list file.
        :param file_path: Path to the surname list.
        :return: List of strings. All the surnames in the file.
        """
        file = open(file_path, 'r')
        names = []
        for i, line in enumerate(file):
            names.append(line[:-1].lower())

        return names

    def __len__(self):
        """
        Inherited from nn.Module. Total number of the data.
        :return: Integer. Length of the data set.
        """
        return self.tags.__len__()

    def __getitem__(self, index):
        """
        Fetch one element from the data set.
        :param index: Index to the element.
        :return: A dict with the following keys:
                'values': Numpy array. The input of the network model.
                'label': Numpy array. The one-hot vectors.
                'raw': List of strings. The original names from the file.
                'raw_label': List of integers. The class labels of all the input names.
        """
        one_hot = np.zeros(3).astype(np.float32)
        one_hot[self.tags[index]] = 1
        asciis = [ord(c) for c in self.raw_data[index]]
        asciis = np.array(asciis).astype(np.float32)
        if asciis.__len__() > 12:
            asciis = asciis[:12]
        else:
            asciis = np.pad(asciis, (0, 12 - asciis.__len__()), mode='constant', constant_values=0)
        return {
            'values': asciis,
            'label': one_hot,
            'raw': self.raw_data[index],
            'raw_label': self.tags[index]
        }
