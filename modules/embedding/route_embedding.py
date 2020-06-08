import numpy as np

from dataaccess.simulation import numpy_to_tensor
from modules.embedding.base_embedding import BaseEmbedding


class RouteEmbedding(BaseEmbedding):
    def __init__(self):
        super(RouteEmbedding, self).__init__()

    def embed(self, input_sequences, cuda_flag, eval_mode):
        embedded_inputs = [input_sequence.route[:] for input_sequence in input_sequences]
        max_node_cnt = max([len(embedded_input) for embedded_input in embedded_inputs])

        for embedded_input in embedded_inputs:
            embedded_input += (max_node_cnt - len(embedded_input))*[0]

        embedded_inputs_array = np.array(embedded_inputs)
        embedded_inputs_tensor = numpy_to_tensor(embedded_inputs_array, 'float', cuda_flag, eval_mode)

        return embedded_inputs_tensor

