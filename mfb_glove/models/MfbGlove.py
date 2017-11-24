import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import sys
sys.path.append("..")


class MfbGlove(nn.Module):
    def __init__(self, opt):
        super(MfbGlove, self).__init__()
        self.opt = opt
        self.JOINT_EMB_SIZE = opt.MFB_FACTOR_NUM * opt.MFB_OUT_DIM
        self.Embedding = nn.Embedding(opt.quest_vob_size, 300)
        self.LSTM = nn.LSTM(input_size=300*2, hidden_size=opt.LSTM_UNIT_NUM, num_layers=1, batch_first=False)
        self.Linear_dataproj = nn.Linear(opt.LSTM_UNIT_NUM, self.JOINT_EMB_SIZE)
        self.Linear_imgproj = nn.Linear(opt.IMAGE_CHANNEL, self.JOINT_EMB_SIZE)
        self.Linear_predict = nn.Linear(opt.MFB_OUT_DIM, opt.NUM_OUTPUT_UNITS)

    def forward(self, data, word_length, img_feature, glove, mode):
        if mode == 'val':
            self.batch_size = self.opt.VAL_BATCH_SIZE
        else:
            self.batch_size = self.opt.BATCH_SIZE
        data_out = Variable(torch.zeros(self.batch_size, self.opt.LSTM_UNIT_NUM)).cuda()
        data = torch.transpose(data, 1, 0).long() 
        glove = glove.permute(1, 0, 2)
        data = F.tanh(self.Embedding(data))                         # T,N,300 
        #glove
        concat_embed = torch.cat((data, glove.float()), 2)                  # T,N,600
        data_lstm, _ = self.LSTM(concat_embed) 
        for i in range(self.batch_size):
            data_out[i] = data_lstm[int(word_length[i]) - 1][i]
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
