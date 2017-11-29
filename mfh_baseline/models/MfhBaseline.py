import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import sys
sys.path.append("..")


class MfhBaseline(nn.Module):
    def __init__(self, opt):
        super(MfhBaseline, self).__init__()
        self.opt = opt
        self.JOINT_EMB_SIZE = opt.MFB_FACTOR_NUM * opt.MFB_OUT_DIM
        self.Embedding = nn.Embedding(opt.quest_vob_size, 300)
        self.LSTM1 = nn.LSTM(input_size=300, hidden_size=opt.LSTM_UNIT_NUM, num_layers=1, batch_first=False)
        # self.LSTM2 = nn.LSTM(input_size=opt.LSTM_UNIT_NUM, hidden_size=opt.LSTM_UNIT_NUM, num_layers=1, batch_first=False)
        self.Linear_dataproj1 = nn.Linear(opt.LSTM_UNIT_NUM, self.JOINT_EMB_SIZE)
        self.Linear_dataproj2 = nn.Linear(opt.LSTM_UNIT_NUM, self.JOINT_EMB_SIZE)
        self.Linear_imgproj1 = nn.Linear(opt.IMAGE_CHANNEL, self.JOINT_EMB_SIZE)
        self.Linear_imgproj2 = nn.Linear(opt.IMAGE_CHANNEL, self.JOINT_EMB_SIZE)
        self.Linear_predict = nn.Linear(opt.MFB_OUT_DIM * 2, opt.NUM_OUTPUT_UNITS)
        self.Dropout1 = nn.Dropout(p=opt.LSTM_DROPOUT_RATIO)
        self.Dropout2 = nn.Dropout(p=opt.MFB_DROPOUT_RATIO)

    def forward(self, data, word_length, img_feat, mode):
        if mode == 'val':
            self.batch_size = self.opt.VAL_BATCH_SIZE
        else:
            self.batch_size = self.opt.BATCH_SIZE
        data_out = Variable(torch.zeros(self.batch_size, self.opt.LSTM_UNIT_NUM)).cuda()
        # """sort(desc)"""
        # word_length, sort_idx = torch.sort(word_length, dim=0, descending=True)
        # _, unsort_idx = torch.sort(sort_idx)
        # data = data[sort_idx]
        # data = torch.transpose(data, 1, 0).long()                           # N x T -> T x N
        # """order 1"""
        # data = F.tanh(self.Embedding(data))                                 # (15,N,300)
        # """pack"""
        # data = nn.utils.rnn.pack_padded_sequence(data, word_length.cpu().numpy(), batch_first=False)
        # data, _ = self.LSTM1(data)                                          # (15,N,1024)
        # """unpack"""
        # data = nn.utils.rnn.pad_packed_sequence(data, batch_first=False)
        # data = data[0]                                                      # get output 15,N,1024
        # for i in range(self.batch_size):
        #     data_out_1[i]=data[int(word_length[i]) - 1][i]                  # get output N,1024
        # """unsort"""
        # data_out_1 = data_out_1[unsort_idx]                                 # N x 1024
        # q_feat = self.Dropout1(data_out_1)
        #######################################
        data = torch.transpose(data, 1, 0).long() 
        data = F.tanh(self.Embedding(data)) 
        data, _ = self.LSTM1(data) 
        for i in range(self.batch_size):
            data_out[i] = data[int(word_length[i]) - 1][i]
        q_feat = self.Dropout1(data_out)

        # data = F.dropout(data, self.opt.LSTM_DROPOUT_RATIO)
        # """order 2"""
        # data = nn.utils.rnn.pack_padded_sequence(data, word_length, batch_first=False)
        # data, _ = self.LSTM2(data)                                   # (15,N,1024)
        # data = nn.utils.rnn.pad_packed_sequence(data, batch_first=False)
        # data = data[0]                                              # get output 15,N,1024
        # for i in range(self.batch_size):
        #     data_out_2[i]=data[int(word_length[i]) - 1][i]          # get output N,1024
        # data_out_2 = F.dropout(data_out_2, self.opt.LSTM_DROPOUT_RATIO)
        # """unsort"""
        # data_out_1 = data_out_1[torch.LongTensor(unsort_idx).cuda()]
        # data_out_2 = data_out_2[torch.LongTensor(unsort_idx).cuda()]          
        # q_feat = torch.cat((data_out_1, data_out_2), 1)             # N,1024*2

        mfb_q_o2_proj = self.Linear_dataproj1(q_feat)                       # data_out (N, 5000)
        mfb_i_o2_proj = self.Linear_imgproj1(img_feat.float())              # img_feature (N, 5000)
        mfb_iq_o2_eltwise = torch.mul(mfb_q_o2_proj, mfb_i_o2_proj)
        mfb_iq_o2_drop = self.Dropout2(mfb_iq_o2_eltwise)
        mfb_iq_o2_resh = mfb_iq_o2_drop.view(-1, 1, self.opt.MFB_OUT_DIM, self.opt.MFB_FACTOR_NUM)  # N x 1 x 1000 x 5
        mfb_o2_out = torch.squeeze(torch.sum(mfb_iq_o2_resh, 3))                            # N x 1000
        mfb_o2_out = torch.sqrt(F.relu(mfb_o2_out)) - torch.sqrt(F.relu(-mfb_o2_out))       # signed sqrt
        #l2 norm
        # mfb_o2_out_l2 = torch.sqrt(torch.sum(torch.pow(mfb_o2_out, 2), 1) + 1e-7)
        # mfb_o2_out = torch.div(mfb_o2_out, torch.unsqueeze(mfb_o2_out_l2, 1).expand(self.batch_size, self.opt.MFB_OUT_DIM))
        mfb_o2_out = F.normalize(mfb_o2_out)

        mfb_q_o3_proj = self.Linear_dataproj2(q_feat)                   # data_out (N, 5000)
        mfb_i_o3_proj = self.Linear_imgproj2(img_feat.float())          # img_feature (N, 5000)
        mfb_iq_o3_eltwise = torch.mul(mfb_q_o3_proj, mfb_i_o3_proj)
        mfb_iq_o3_eltwise = torch.mul(mfb_iq_o3_eltwise, mfb_iq_o2_drop)
        mfb_iq_o3_drop = self.Dropout2(mfb_iq_o3_eltwise)
        mfb_iq_o3_resh = mfb_iq_o3_drop.view(-1, 1, self.opt.MFB_OUT_DIM, self.opt.MFB_FACTOR_NUM)
        mfb_o3_out = torch.squeeze(torch.sum(mfb_iq_o3_resh, 3))                            # N x 1000
        mfb_o3_out = torch.sqrt(F.relu(mfb_o3_out)) - torch.sqrt(F.relu(-mfb_o3_out))
        #l2 norm
        # mfb_o3_out_l2 = torch.sqrt(torch.sum(torch.pow(mfb_o3_out, 2), 1) + 1e-7)
        # mfb_o3_out = torch.div(mfb_o3_out, torch.unsqueeze(mfb_o3_out_l2, 1).expand(self.batch_size, self.opt.MFB_OUT_DIM))
        mfb_o3_out = F.normalize(mfb_o3_out)

        mfb_o23_out = torch.cat((mfb_o2_out, mfb_o3_out), 1)        #200,2000     
        prediction = self.Linear_predict(mfb_o23_out)               

        prediction = F.log_softmax(prediction)

        return prediction
