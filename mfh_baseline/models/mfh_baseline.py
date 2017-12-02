import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import sys
sys.path.append("..")


class mfh_baseline(nn.Module):
    def __init__(self, opt):
        super(mfh_baseline, self).__init__()
        self.opt = opt
        self.JOINT_EMB_SIZE = opt.MFB_FACTOR_NUM * opt.MFB_OUT_DIM
        self.Embedding = nn.Embedding(opt.quest_vob_size, 300)
        self.LSTM1 = nn.LSTM(input_size=300, hidden_size=opt.LSTM_UNIT_NUM, num_layers=1, batch_first=False)
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
        data = torch.transpose(data, 1, 0).long() 
        data = F.tanh(self.Embedding(data)) 
        data, _ = self.LSTM1(data) 
        for i in range(self.batch_size):
            data_out[i] = data[int(word_length[i]) - 1][i]
        q_feat = self.Dropout1(data_out)

        mfb_q_o2_proj = self.Linear_dataproj1(q_feat)                       # data_out (N, 5000)
        mfb_i_o2_proj = self.Linear_imgproj1(img_feat.float())              # img_feature (N, 5000)
        mfb_iq_o2_eltwise = torch.mul(mfb_q_o2_proj, mfb_i_o2_proj)
        mfb_iq_o2_drop = self.Dropout2(mfb_iq_o2_eltwise)
        mfb_iq_o2_resh = mfb_iq_o2_drop.view(-1, 1, self.opt.MFB_OUT_DIM, self.opt.MFB_FACTOR_NUM)  # N x 1 x 1000 x 5
        mfb_o2_out = torch.squeeze(torch.sum(mfb_iq_o2_resh, 3))                            # N x 1000
        mfb_o2_out = torch.sqrt(F.relu(mfb_o2_out)) - torch.sqrt(F.relu(-mfb_o2_out))       # signed sqrt
        mfb_o2_out = F.normalize(mfb_o2_out)

        mfb_q_o3_proj = self.Linear_dataproj2(q_feat)                   # data_out (N, 5000)
        mfb_i_o3_proj = self.Linear_imgproj2(img_feat.float())          # img_feature (N, 5000)
        mfb_iq_o3_eltwise = torch.mul(mfb_q_o3_proj, mfb_i_o3_proj)
        mfb_iq_o3_eltwise = torch.mul(mfb_iq_o3_eltwise, mfb_iq_o2_drop)
        mfb_iq_o3_drop = self.Dropout2(mfb_iq_o3_eltwise)
        mfb_iq_o3_resh = mfb_iq_o3_drop.view(-1, 1, self.opt.MFB_OUT_DIM, self.opt.MFB_FACTOR_NUM)
        mfb_o3_out = torch.squeeze(torch.sum(mfb_iq_o3_resh, 3))                            # N x 1000
        mfb_o3_out = torch.sqrt(F.relu(mfb_o3_out)) - torch.sqrt(F.relu(-mfb_o3_out))
        mfb_o3_out = F.normalize(mfb_o3_out)

        mfb_o23_out = torch.cat((mfb_o2_out, mfb_o3_out), 1)        #200,2000     
        prediction = self.Linear_predict(mfb_o23_out)               
        prediction = F.log_softmax(prediction)

        return prediction
