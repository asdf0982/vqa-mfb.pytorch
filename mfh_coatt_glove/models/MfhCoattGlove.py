import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import sys
sys.path.append("..")


class MfhCoattGlove(nn.Module):
    def __init__(self, opt):
        super(MfhCoattGlove, self).__init__()
        self.opt = opt
        self.JOINT_EMB_SIZE = opt.MFB_FACTOR_NUM * opt.MFB_OUT_DIM
        self.Embedding = nn.Embedding(opt.quest_vob_size, 300)
        self.LSTM = nn.LSTM(input_size=300*2, hidden_size=opt.LSTM_UNIT_NUM, num_layers=1, batch_first=False)
        self.Softmax = nn.Softmax()

        self.Linear1_q_proj = nn.Linear(opt.LSTM_UNIT_NUM*opt.NUM_QUESTION_GLIMPSE, self.JOINT_EMB_SIZE)
        self.Linear2_q_proj = nn.Linear(opt.LSTM_UNIT_NUM*opt.NUM_QUESTION_GLIMPSE, self.JOINT_EMB_SIZE)
        self.Linear3_q_proj = nn.Linear(opt.LSTM_UNIT_NUM*opt.NUM_QUESTION_GLIMPSE, self.JOINT_EMB_SIZE)
        self.Conv1_i_proj = nn.Conv2d(opt.IMAGE_CHANNEL, self.JOINT_EMB_SIZE, 1)
        self.Linear2_i_proj = nn.Linear(opt.IMAGE_CHANNEL*opt.NUM_IMG_GLIMPSE, self.JOINT_EMB_SIZE)
        self.Linear3_i_proj = nn.Linear(opt.IMAGE_CHANNEL*opt.NUM_IMG_GLIMPSE, self.JOINT_EMB_SIZE)

        self.Dropout_L = nn.Dropout(p=opt.LSTM_DROPOUT_RATIO)
        self.Dropout_M = nn.Dropout(p=opt.MFB_DROPOUT_RATIO)
        self.Conv1_Qatt = nn.Conv2d(1024, 512, 1)
        self.Conv2_Qatt = nn.Conv2d(512, opt.NUM_QUESTION_GLIMPSE, 1)
        self.Conv1_Iatt = nn.Conv2d(1000, 512, 1)
        self.Conv2_Iatt = nn.Conv2d(512, opt.NUM_IMG_GLIMPSE, 1)

        self.Linear_predict = nn.Linear(opt.MFB_OUT_DIM*2, opt.NUM_OUTPUT_UNITS)


    def forward(self, data, word_length, img_feature, glove, mode):
        if mode == 'val' or mode == 'test' or mode == 'test-dev':
            self.batch_size = self.opt.VAL_BATCH_SIZE
        else:
            self.batch_size = self.opt.BATCH_SIZE
        lstm1_pad_pad = Variable(torch.zeros(self.opt.MAX_WORDS_IN_QUESTION, self.batch_size, self.opt.LSTM_UNIT_NUM)).cuda()
        """q sort p"""    
        word_length, sort_idx = torch.sort(word_length, dim=0, descending=True)
        _, unsort_idx = torch.sort(sort_idx)
        data = data[sort_idx]                                       # N x T
        glove = glove[sort_idx]                                     # N x T x 300
        """d sort b"""
        data = torch.transpose(data, 1, 0)                          # type Longtensor,  T x N 
        glove = glove.permute(1, 0, 2)                              # type float, T x N x 300
        embed_tanh= F.tanh(self.Embedding(data))                    # T x N x 300
        concat_word_embed = torch.cat((embed_tanh, glove), 2)       # T x N x 600
        concat_word_embed_pack = nn.utils.rnn.pack_padded_sequence(concat_word_embed, word_length.cpu().numpy(), batch_first=False)
        lstm1, _ = self.LSTM(concat_word_embed_pack)                     # T x N x 1024
        lstm1_pad = nn.utils.rnn.pad_packed_sequence(lstm1, batch_first=False) # (max lenth in each batch x N x 1024)
        lstm1_pad = lstm1_pad[0]
        batch_max_len = lstm1_pad.size()[0]
        need_len = self.opt.MAX_WORDS_IN_QUESTION - batch_max_len
        if need_len!=0:
            temp_pad = Variable(torch.zeros(need_len, self.batch_size, self.opt.LSTM_UNIT_NUM)).cuda()
            lstm1_pad = torch.cat((lstm1_pad, temp_pad), dim=0)     # T x N x 1024
        """unsort"""
        lstm1_resh = lstm1_pad.permute(1, 2, 0)                     # N x 1024 x T
        lstm1_unsort = lstm1_resh[unsort_idx]
        lstm1_droped = self.Dropout_L(lstm1_unsort)
        lstm1_resh2 = torch.unsqueeze(lstm1_droped, 3)              # N x 1024 x T x 1
        '''
        Question Attention
        '''        
        qatt_conv1 = self.Conv1_Qatt(lstm1_resh2)                   # N x 512 x T x 1
        qatt_relu = F.relu(qatt_conv1)
        qatt_conv2 = self.Conv2_Qatt(qatt_relu)                     # N x 2 x T x 1
        qatt_conv2 = qatt_conv2.view(self.batch_size*self.opt.NUM_QUESTION_GLIMPSE,-1)
        qatt_softmax = self.Softmax(qatt_conv2)
        qatt_softmax = qatt_softmax.view(self.batch_size, self.opt.NUM_QUESTION_GLIMPSE, -1, 1)
        qatt_feature_list = []
        for i in range(self.opt.NUM_QUESTION_GLIMPSE):
            t_qatt_mask = qatt_softmax.narrow(1, i, 1)              # N x 1 x T x 1
            t_qatt_mask = t_qatt_mask * lstm1_resh2                 # N x 1024 x T x 1
            t_qatt_mask = torch.sum(t_qatt_mask, 2, keepdim=True)   # N x 1024 x 1 x 1
            qatt_feature_list.append(t_qatt_mask)
        qatt_feature_concat = torch.cat(qatt_feature_list, 1)       # N x 2048 x 1 x 1
        '''
        Image Attention with MFB
        '''
        q_feat_resh = torch.squeeze(qatt_feature_concat)                                # N x 2048
        i_feat_resh = torch.unsqueeze(img_feature, 3)                                   # N x 2048 x 100 x 1
        iatt_q_proj = self.Linear1_q_proj(q_feat_resh)                                  # N x 5000
        iatt_q_resh = iatt_q_proj.view(self.batch_size, self.JOINT_EMB_SIZE, 1, 1)      # N x 5000 x 1 x 1
        # iatt_q_tile = iatt_q_resh.expand(self.batch_size, self.JOINT_EMB_SIZE, 100, 1)  # N x 5000 x 100 x 1
        iatt_i_conv = self.Conv1_i_proj(i_feat_resh)                                     # N x 5000 x 100 x 1
        # iatt_iq_eltwise = torch.mul(iatt_q_tile, iatt_i_conv)                           # N x 5000 x 100 x 1
        iatt_iq_eltwise = iatt_q_resh * iatt_i_conv
        iatt_iq_droped = self.Dropout_M(iatt_iq_eltwise)                                # N x 5000 x 100 x 1
        iatt_iq_permute1 = iatt_iq_droped.permute(0,2,1,3).contiguous()                 # N x 100 x 5000 x 1
        iatt_iq_resh = iatt_iq_permute1.view(self.batch_size, self.opt.IMG_FEAT_SIZE, self.opt.MFB_OUT_DIM, self.opt.MFB_FACTOR_NUM)
        iatt_iq_sumpool = torch.sum(iatt_iq_resh, 3, keepdim=True)                      # N x 100 x 1000 x 1 
        iatt_iq_permute2 = iatt_iq_sumpool.permute(0,2,1,3)                             # N x 1000 x 100 x 1
        iatt_iq_sqrt = torch.sqrt(F.relu(iatt_iq_permute2)) - torch.sqrt(F.relu(-iatt_iq_permute2))
        iatt_iq_sqrt = iatt_iq_sqrt.view(self.batch_size, -1)                           # N x 100000
        iatt_iq_l2 = F.normalize(iatt_iq_sqrt)
        iatt_iq_l2 = iatt_iq_l2.view(self.batch_size, self.opt.MFB_OUT_DIM, self.opt.IMG_FEAT_SIZE, 1)  # N x 1000 x 100 x 1

        ## 2 conv layers 1000 -> 512 -> 2
        iatt_conv1 = self.Conv1_Iatt(iatt_iq_l2)                    # N x 512 x 100 x 1
        iatt_relu = F.relu(iatt_conv1)
        iatt_conv2 = self.Conv2_Iatt(iatt_relu)                     # N x 2 x 100 x 1
        iatt_conv2 = iatt_conv2.view(self.batch_size*self.opt.NUM_IMG_GLIMPSE, -1)
        iatt_softmax = self.Softmax(iatt_conv2)
        iatt_softmax = iatt_softmax.view(self.batch_size, self.opt.NUM_IMG_GLIMPSE, -1, 1)
        iatt_feature_list = []
        for i in range(self.opt.NUM_IMG_GLIMPSE):
            t_iatt_mask = iatt_softmax.narrow(1, i, 1)              # N x 1 x 100 x 1
            t_iatt_mask = t_iatt_mask * i_feat_resh                 # N x 2048 x 100 x 1
            t_iatt_mask = torch.sum(t_iatt_mask, 2, keepdim=True)   # N x 2048 x 1 x 1
            iatt_feature_list.append(t_iatt_mask)
        iatt_feature_concat = torch.cat(iatt_feature_list, 1)       # N x 4096 x 1 x 1
        iatt_feature_concat = torch.squeeze(iatt_feature_concat)    # N x 4096
        '''
        Fine-grained Image-Question MFH fusion
        '''
        mfb_q_o2_proj = self.Linear2_q_proj(q_feat_resh)               # N x 5000
        mfb_i_o2_proj = self.Linear2_i_proj(iatt_feature_concat)        # N x 5000
        mfb_iq_o2_eltwise = torch.mul(mfb_q_o2_proj, mfb_i_o2_proj)          # N x 5000
        mfb_iq_o2_drop = self.Dropout_M(mfb_iq_o2_eltwise)
        mfb_iq_o2_resh = mfb_iq_o2_drop.view(self.batch_size, 1, self.opt.MFB_OUT_DIM, self.opt.MFB_FACTOR_NUM)   # N x 1 x 1000 x 5
        mfb_iq_o2_sumpool = torch.sum(mfb_iq_o2_resh, 3, keepdim=True)    # N x 1 x 1000 x 1
        mfb_o2_out = torch.squeeze(mfb_iq_o2_sumpool)                     # N x 1000
        mfb_o2_sign_sqrt = torch.sqrt(F.relu(mfb_o2_out)) - torch.sqrt(F.relu(-mfb_o2_out))
        mfb_o2_l2 = F.normalize(mfb_o2_sign_sqrt)

        mfb_q_o3_proj = self.Linear3_q_proj(q_feat_resh)               # N x 5000
        mfb_i_o3_proj = self.Linear3_i_proj(iatt_feature_concat)        # N x 5000
        mfb_iq_o3_eltwise = torch.mul(mfb_q_o3_proj, mfb_i_o3_proj)          # N x 5000
        mfb_iq_o3_drop = self.Dropout_M(mfb_iq_o3_eltwise)
        mfb_iq_o3_resh = mfb_iq_o3_drop.view(self.batch_size, 1, self.opt.MFB_OUT_DIM, self.opt.MFB_FACTOR_NUM)   # N x 1 x 1000 x 5
        mfb_iq_o3_sumpool = torch.sum(mfb_iq_o3_resh, 3, keepdim=True)    # N x 1 x 1000 x 1
        mfb_o3_out = torch.squeeze(mfb_iq_o3_sumpool)                     # N x 1000
        mfb_o3_sign_sqrt = torch.sqrt(F.relu(mfb_o3_out)) - torch.sqrt(F.relu(-mfb_o3_out))
        mfb_o3_l2 = F.normalize(mfb_o3_sign_sqrt)

        mfb_o23_l2 = torch.cat((mfb_o2_l2, mfb_o3_l2), 1)               # N x 2000
        prediction = self.Linear_predict(mfb_o23_l2)
        prediction = F.log_softmax(prediction)

        return prediction