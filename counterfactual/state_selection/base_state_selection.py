class BaseStateSelection:

    def __init__(self, dataset):
        self.dataset = dataset
        pass

    def select_indices(self, n):
        raise NotImplementedError