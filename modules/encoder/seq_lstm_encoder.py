import torch.nn as nn

from modules.encoder.base_encoder import BaseEncoder


class SeqLstmEncoder(BaseEncoder):
    def __init__(self, args):
        super(SeqLstmEncoder, self).__init__(args)
        self.hidden_size = args.LSTM_hidden_size
        self.embedding_size = args.embedding_size
        self.num_layers = args.num_LSTM_layers
        self.dropout_rate = args.dropout_rate
        self.encoder = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size,
                               num_layers=self.num_layers, batch_first=True, dropout=self.dropout_rate,
                               bidirectional=True)

    def encode(self, raw_input_list, embedded_input_list):
        encoder_output, encoder_state = self.encoder(embedded_input_list)

        for idx, item in enumerate(raw_input_list):
            item.encoder_outputs = encoder_output[idx]

        return raw_input_list
