import torch.utils.data as Data

class BaseDataset(Data.Dataset):
    """
    A general base class for different datasets.
    """

    def __init__(self, split, config):
        self.config = config
        self.split = split
        self.files = []
        self.length = 0

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        This function should be implemented in the child classes.
        """
        raise NotImplementedError("Child classes should implement this method.")