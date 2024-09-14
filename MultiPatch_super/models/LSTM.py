
import torch
from torch import nn
class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.input_size = configs.enc_in
        self.pred_len = configs.pred_len
        self.hidden_size = configs.d_model
        self.num_layers = configs.e_layers#层数
        self.output_size = configs.c_out
        self.num_directions = 1 #单向
        self.batch_size = configs.batch_size
        # num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs 
        # together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing 
        # the final results. Default: 1
        self.lstm = nn.LSTM((self.input_size), (self.hidden_size), (self.num_layers), batch_first=True)#将batch_size提前
        self.Linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        device = torch.device('cuda')
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]#输入到lstm中的input的shape应该 input(batch_size,seq_len,input_dim)
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        output, _ = self.lstm(input_seq, (h_0, c_0))    # (隐状态h_n，单元状态c_n)
        pred = self.Linear(output)
        pred = pred[:, -self.pred_len:, :]
        return pred