import numpy as np

from dataaccess.data_parser import VrpDataParser


class VrpDataProcessor:
    def __init__(self):
        self.parser = VrpDataParser()

    def get_batch(self, data, batch_size, start_idx):
        data_size = len(data)
        if start_idx is not None:
            batch_idxes = [i for i in range(start_idx, min(data_size, start_idx + batch_size))]
        else:
            batch_idxes = np.random.choice(len(data), batch_size)
        batch_data = []
        for idx in batch_idxes:
            problem = data[idx]
            dm = self.parser.parse(problem)
            batch_data.append(dm)
        return batch_data

