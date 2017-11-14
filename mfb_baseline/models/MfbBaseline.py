import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import sys
sys.path.append("..")


class MfbBaseline(nn.Module):
    def __init__(self, opt):
        super(MfbBaseline, self).__init__()
        self.opt = opt
        self.JOINT_EMB_SIZE = opt.MFB_FACTOR_NUM * opt.MFB_OUT_DIM
        self.Embedding = nn.Embedding(opt.quest_vob_size, 300)
        self.LSTM = nn.LSTM(input_size=300, hidden_size=opt.LSTM_UNIT_NUM, num_layers=1, batch_first=False)
        self.Linear_dataproj = nn.Linear(opt.LSTM_UNIT_NUM, self.JOINT_EMB_SIZE)
        self.Linear_imgproj = nn.Linear(opt.IMAGE_CHANNEL, self.JOINT_EMB_SIZE)
        self.Linear_predict = nn.Linear(opt.MFB_OUT_DIM, opt.NUM_OUTPUT_UNITS)

    def forward(self, data, word_length, img_feature, mode):
        if mode == 'val':
            self.batch_size = self.opt.VAL_BATCH_SIZE
        else:
            self.batch_size = self.opt.BATCH_SIZE
        data_out = Variable(torch.zeros(self.batch_size, self.opt.LSTM_UNIT_NUM)).cuda()
        """sort(desc)"""
        word_length, sort_idx = torch.sort(word_length, dim=0, descending=True)
        _, unsort_idx = torch.sort(sort_idx)
        data = data[sort_idx]
        data = torch.transpose(data, 1, 0).long()                           # N x T -> T x N
        data = F.tanh(self.Embedding(data))                                 # (15,batch,300)
        """pack"""
        data = nn.utils.rnn.pack_padded_sequence(data, word_length.cpu().numpy(), batch_first=False)
        data, _ = self.LSTM(data)                                           # (15,batch,1024)
        """unpack"""
        data = nn.utils.rnn.pad_packed_sequence(data, batch_first=False)
        data = data[0]                                                      # get output 15,200,1024
        for i in range(self.batch_size):
            data_out[i]=data[int(word_length[i]) - 1][i]
        """unsort"""
        data_out = data_out[unsort_idx]
        # word_length = word_length[unsort_idx]
        data_out = F.dropout(data_out, self.opt.LSTM_DROPOUT_RATIO, training=self.training)
        data_out = self.Linear_dataproj(data_out)                   # data_out (batch, 5000)
        img_feature = self.Linear_imgproj(img_feature.float())      # img_feature (batch, 5000)
        iq = torch.mul(data_out, img_feature)
        iq = F.dropout(iq, self.opt.MFB_DROPOUT_RATIO, training=self.training)
        iq = iq.view(-1, 1, self.opt.MFB_OUT_DIM, self.opt.MFB_FACTOR_NUM)
        iq = torch.squeeze(torch.sum(iq, 3))                        # sum pool
        iq = torch.sqrt(F.relu(iq)) - torch.sqrt(F.relu(-iq))       # signed sqrt
        iq = F.normalize(iq)
        iq = self.Linear_predict(iq)                                # (64,3000)
        iq = F.log_softmax(iq)
        
        return iq
