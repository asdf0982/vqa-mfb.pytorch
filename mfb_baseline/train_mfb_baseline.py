# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import os
import sys
import config
from models.MfbBaseline import MfbBaseline
import utils.data_provider as data_provider
from utils.data_provider import VQADataProvider
from utils.eval_utils import exec_validation, drawgraph
import json
import datetime
from tensorboardX import SummaryWriter 
sys.path.append(config.VQA_TOOLS_PATH)
sys.path.append(config.VQA_EVAL_TOOLS_PATH)
from vqaTools.vqa import VQA
from vqaEvaluation.vqaEval import VQAEval

def make_answer_vocab(adic, vocab_size):
    """
    Returns a dictionary that maps words to indices.
    """
    adict = {'':0}
    nadict = {'':1000000}
    vid = 1
    for qid in adic.keys():
        answer_obj = adic[qid]
        answer_list = [ans['answer'] for ans in answer_obj]
        
        for q_ans in answer_list:
            # create dict
            if adict.has_key(q_ans):
                nadict[q_ans] += 1
            else:
                nadict[q_ans] = 1
                adict[q_ans] = vid
                vid +=1

    # debug
    nalist = []
    for k,v in sorted(nadict.items(), key=lambda x:x[1]):
        nalist.append((k,v))

    # remove words that appear less than once 
    n_del_ans = 0
    n_valid_ans = 0
    adict_nid = {}
    for i, w in enumerate(nalist[:-vocab_size]):
        del adict[w[0]]
        n_del_ans += w[1]
    for i, w in enumerate(nalist[-vocab_size:]):
        n_valid_ans += w[1]
        adict_nid[w[0]] = i
    
    return adict_nid

def make_question_vocab(qdic):
    """
    Returns a dictionary that maps words to indices.
    """
    vdict = {'':0}
    vid = 1
    for qid in qdic.keys():
        # sequence to list
        q_str = qdic[qid]['qstr']
        q_list = VQADataProvider.seq_to_list(q_str)

        # create dict
        for w in q_list:
            if not vdict.has_key(w):
                vdict[w] = vid
                vid +=1

    return vdict

def make_vocab_files():
    """
    Produce the question and answer vocabulary files.
    """
    print ('making question vocab...', opt.QUESTION_VOCAB_SPACE)
    qdic, _ = VQADataProvider.load_data(opt.QUESTION_VOCAB_SPACE)
    question_vocab = make_question_vocab(qdic)
    print ('making answer vocab...', opt.ANSWER_VOCAB_SPACE)
    _, adic = VQADataProvider.load_data(opt.ANSWER_VOCAB_SPACE)
    answer_vocab = make_answer_vocab(adic, opt.NUM_OUTPUT_UNITS)
    return question_vocab, answer_vocab

def adjust_learning_rate(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

def train():
    criterion = nn.KLDivLoss(size_average=False)
    train_loss = np.zeros(opt.MAX_ITERATIONS + 1)
    results = []
    for iter_idx, (data, word_length, feature, answer, epoch) in enumerate(train_Loader):
        model.train()
        data = np.squeeze(data, axis=0)
        word_length = np.squeeze(word_length, axis=0)
        feature = np.squeeze(feature, axis=0)
        answer = np.squeeze(answer, axis=0)
        epoch = epoch.numpy()

        data = Variable(data).cuda()
        word_length = word_length.cuda()
        img_feature = Variable(feature).cuda()
        label = Variable(answer).cuda().float()
        optimizer.zero_grad()
        pred = model(data, word_length, img_feature, 'train')
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        train_loss[iter_idx] = loss.data[0]
        if iter_idx % opt.DECAY_STEPS == 0 and iter_idx != 0:
            adjust_learning_rate(optimizer, opt.DECAY_RATE)
        if iter_idx % opt.PRINT_INTERVAL == 0 and iter_idx != 0:
            now = str(datetime.datetime.now())
            c_mean_loss = train_loss[iter_idx-opt.PRINT_INTERVAL:iter_idx].mean()/opt.BATCH_SIZE
            writer.add_scalar('mfb_baseline/train_loss', c_mean_loss, iter_idx)
            writer.add_scalar('mfb_baseline/lr', optimizer.param_groups[0]['lr'], iter_idx)
            print('{}\tTrain Epoch: {}\tIter: {}\tLoss: {:.4f}'.format(
                        now, epoch, iter_idx, c_mean_loss))
        if iter_idx % opt.CHECKPOINT_INTERVAL == 0 and iter_idx != 0:
            if not os.path.exists('./data'):
                os.makedirs('./data')
            save_path = './data/mfb_baseline_iter_' + str(iter_idx) + '.pth'
            torch.save(model.state_dict(), save_path)
        if iter_idx % opt.VAL_INTERVAL == 0 and iter_idx != 0:
            test_loss, acc_overall, acc_per_ques, acc_per_ans = exec_validation(model, opt, mode='val', folder=folder, it=iter_idx)
            writer.add_scalar('mfb_baseline/val_loss', test_loss, iter_idx)
            writer.add_scalar('mfb_baseline/accuracy', acc_overall, iter_idx)
            print ('Test loss:', test_loss)
            print ('Accuracy:', acc_overall)
            print ('Test per ans', acc_per_ans)
            results.append([iter_idx, c_mean_loss, test_loss, acc_overall, acc_per_ques, acc_per_ans])
            best_result_idx = np.array([x[3] for x in results]).argmax()
            print ('Best accuracy of', results[best_result_idx][3], 'was at iteration', results[best_result_idx][0])
            drawgraph(results, folder, opt.MFB_FACTOR_NUM, opt.MFB_OUT_DIM, prefix='mfb_baseline')
        if iter_idx % opt.TESTDEV_INTERVAL == 0 and iter_idx != 0:
            exec_validation(model, opt, mode='test-dev', folder=folder, it=iter_idx)

opt = config.parse_opt()
torch.cuda.set_device(opt.TRAIN_GPU_ID)
# torch.cuda.manual_seed(opt.SEED)
writer = SummaryWriter()
folder = 'mfb_baseline_%s'%opt.TRAIN_DATA_SPLITS
if not os.path.exists('./%s'%folder):
    os.makedirs('./%s'%folder)
question_vocab, answer_vocab = {}, {}
if os.path.exists('./%s/vdict.json'%folder) and os.path.exists('./%s/adict.json'%folder):
    print ('restoring vocab')
    with open('./%s/vdict.json'%folder,'r') as f:
        question_vocab = json.load(f)
    with open('./%s/adict.json'%folder,'r') as f:
        answer_vocab = json.load(f)
else:
    question_vocab, answer_vocab = make_vocab_files()
    with open('./%s/vdict.json'%folder,'w') as f:
        json.dump(question_vocab, f)
    with open('./%s/adict.json'%folder,'w') as f:
        json.dump(answer_vocab, f)
print ('question vocab size:', len(question_vocab))
print ('answer vocab size:', len(answer_vocab))
opt.quest_vob_size = len(question_vocab)
opt.ans_vob_size = len(answer_vocab)

train_Data = data_provider.VQADataset(opt.TRAIN_DATA_SPLITS, opt.BATCH_SIZE, folder, opt)
train_Loader = torch.utils.data.DataLoader(dataset=train_Data, shuffle=True, pin_memory=True, num_workers=1)

model = MfbBaseline(opt)
if opt.RESUME:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(opt.RESUME_PATH)
    model.load_state_dict(checkpoint)
else:
    '''init model parameter'''
    for name, param in model.named_parameters():
        if 'bias' in name:  # bias can't init by xavier
            init.constant(param, 0.0)
        elif 'weight' in name:
            init.kaiming_uniform(param)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=opt.INIT_LERARNING_RATE)

train()
writer.close()