# -*- coding:utf-8 -*-
# @Time    :2023/6/18 10:30
# @Author  :ZZK
# @ File   :my_model.py
# Description:
from torchvision import models
from transformers import BertModel
from torchcrf import CRF

from utils import *
from config import tag2idx, max_len, max_node


class BiLSTM(nn.Module):
    def __init__(self, input_size=768, hidden_size=128, num_layers=2, dropout=0.2):
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size * 2, input_size)

    def forward(self, input_tensor):
        # input_tensor shape: (batch_size, sequence_length, input_size)
        output, _ = self.lstm(input_tensor)
        # output shape: (batch_size, sequence_length, hidden_size * 2)
        output = self.dropout_layer(output)
        # linear transform to match the input tensor shape
        output = self.linear(output)
        return output


class MMNerModel(nn.Module):

    def __init__(self, d_model=512, d_hidden=256, n_heads=8, dropout=0.4, tag2idx=tag2idx):
        super(MMNerModel, self).__init__()  # 调用父类的构造函数

        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.resnet = models.resnet152(pretrained=True)
        self.crf = CRF(len(tag2idx), batch_first=True)
        # self.hidden2tag = nn.Linear(2*d_model, len(tag2idx))
        self.hidden2tag = nn.Linear(in_features=(1280), out_features=len(tag2idx))
        self.hidden2tag_s = nn.Linear(in_features=(512), out_features=len(tag2idx))
        # self.hidden2tag = nn.Linear(in_features=512, out_features=len(tag2idx))

        objcnndim = 2048
        fc_feats = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(
            in_features=fc_feats, out_features=objcnndim, bias=True)

        self.dp = dropout  # 0.4
        self.d_model = d_model  # 512
        self.hid = d_hidden  # 256

        self.trans_txt = nn.Linear(768, d_model)
        self.trans_obj = nn.Sequential(Linear(objcnndim, d_model), nn.ReLU(), nn.Dropout(dropout),
                                       Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout))
        # BiLSTM
        self.bilstm = BiLSTM(768, 128, 2, 0.2)

        # text
        self.mhatt_x = clone(MultiHeadedAttention(
            n_heads, d_model, dropout), 1)  #
        self.ffn_x = clone(PositionwiseFeedForward(d_model, d_hidden), 1)
        self.res4ffn_x = clone(SublayerConnectionv2(d_model, dropout), 1)
        self.res4mes_x = clone(SublayerConnectionv2(d_model, dropout), 1)

        # img
        self.mhatt_o = clone(MultiHeadedAttention(
            n_heads, d_model, dropout, v=0, output=0), 1)
        self.ffn_o = clone(PositionwiseFeedForward(d_model, d_hidden), 1)
        self.res4mes_o = clone(SublayerConnectionv2(d_model, dropout), 1)
        self.res4ffn_o = clone(SublayerConnectionv2(d_model, dropout), 1)

        self.mhatt_x2o = clone(Linear(d_model * 2, d_model), 1)
        self.mhatt_o2x = clone(Linear(d_model * 2, d_model), 1)
        self.xgate = clone(SublayerConnectionv2(d_model, dropout), 1)
        self.ogate = clone(SublayerConnectionv2(d_model, dropout), 1)

    def likelihood(self, x, b_img, inter_matrix, text_mask, tags):
        """
        inter_matrix: batch, max_len, 4
        text_mask: batch, max_len
        """
        batch_size, xn, objn = inter_matrix.size(0), inter_matrix.size(1), inter_matrix.size(2)
        inter_matrix = inter_matrix.unsqueeze(-1)

        crf_mask = x["attention_mask"]
        x_k = {'input_ids': x['input_k_ids'], 'token_type_ids': x['token_type_ids'],
               'attention_mask': x['attention_mask_k']}
        x_x = {'input_ids': x['input_ids'], 'token_type_ids': x['token_type_ids'],
               'attention_mask': x['attention_mask']}
        x = self.bert(**x_x)[0]

        x_k = self.bert(**x_k)[0]
        o = self.resnet(b_img).view(batch_size, max_node, -1)  # [8,4,2048]
        o = self.trans_obj(o)

        bert_x_k = x_k.clone()
        bert_x = x.clone()  # reserve origin bert output (9, 48, 768)
        x_lstm = self.bilstm(x)
        bert_x_k = self.trans_txt(bert_x_k)
        x_lstm = self.trans_txt(x_lstm)
        x = x_lstm
        # Text self-attention: batch, max_len, dim  句子attention
        newx = self.res4mes_x[0](x, self.mhatt_x[0](
            x, x, x, text_mask.unsqueeze(1)))  # T-ATT
        newo = self.res4mes_o[0](o, self.mhatt_o[0](o, o, o, None))  # V-ATT

        # Text to Image Gating
        newx_ep = newx.unsqueeze(2).expand(
            batch_size, max_len, objn, newx.size(-1))
        o_ep = newo.unsqueeze(1).expand(batch_size, xn, objn, o.size(-1))
        x2o_gates = torch.sigmoid(
            self.mhatt_x2o[0](torch.cat((newx_ep, o_ep), -1)))
        x2o = (x2o_gates * inter_matrix * o_ep).sum(2)
        newx = self.xgate[0](newx, x2o)
        newx = self.res4mes_x[0](newx, self.mhatt_x[0](
            newx, bert_x_k, bert_x_k, text_mask.unsqueeze(1)))
        newx = self.res4ffn_x[0](newx, self.ffn_x[0](newx))
        x = torch.cat((bert_x, newx), dim=2)
        x = self.hidden2tag(x)  # 全连接层

        return -self.crf(x, tags, mask=crf_mask)  # 返回一个对数似然值

    def forward(self, x, b_img, inter_matrix, text_mask, tags):
        """
        inter_matrix: batch, max_len（句子的128）, 4 (1填充、0padding的三维tensor)
        text_mask: batch, max_len（128）   （attention_mask，二维，句子长度n个1，其余0padding
        x：包含了句子的input_ids、token_type_ids、attention_mask三个tensor的dic
        """
        # xn：128  objn：4
        batch_size, xn, objn = inter_matrix.size(0), inter_matrix.size(1), inter_matrix.size(
            2)  # inter_matrix的第0、1、2维度大小
        inter_matrix = inter_matrix.unsqueeze(-1)  # 将inter_matrix由（batch，max_len,4）变成（batch，max_len,4，1）
        matrix4obj = torch.transpose(inter_matrix, 1, 2)
        crf_mask = x["attention_mask"]
        x_k = {'input_ids': x['input_k_ids'], 'token_type_ids': x['token_type_ids'],
               'attention_mask': x['attention_mask_k']}
        x_x = {'input_ids': x['input_ids'], 'token_type_ids': x['token_type_ids'],
               'attention_mask': x['attention_mask']}

        x = self.bert(**x_x)[0]
        x_k = self.bert(**x_k)[0]
        o = self.resnet(b_img).view(batch_size, max_node, -1)  # 调整特征维度
        o = self.trans_obj(o)

        bert_x = x.clone()  # reserve origin bert output (9, 48, 768) copy
        bert_x_k = x_k.clone()
        x_lstm = self.bilstm(x)
        # x = self.trans_txt(x)  # 9, 48, 512
        # bert_x_k = self.trans_txt(bert_x_k)
        x_lstm = self.trans_txt(x_lstm)  # 特征映射-----------------------------------
        x = x_lstm

        # Text self-attention: batch, max_len, dim  文本自注意力
        newx = self.res4mes_x[0](x, self.mhatt_x[0](
            x, x, x, text_mask.unsqueeze(1)))

        newo = self.res4mes_o[0](o, self.mhatt_o[0](o, o, o, None))

        # Text to Image Gating      文本与视觉特征融合（交叉融合）
        newx_ep = newx.unsqueeze(2).expand(
            batch_size, max_len, objn, newx.size(-1))
        o_ep = newo.unsqueeze(1).expand(batch_size, xn, objn, o.size(-1))
        # batch, xn, objn, dmodel
        x2o_gates = torch.sigmoid(
            self.mhatt_x2o[0](torch.cat((newx_ep, o_ep), -1)))
        x2o = (x2o_gates * inter_matrix * o_ep).sum(2)

        newx = self.xgate[0](newx, x2o)  # 句子FFN
        newx = self.res4mes_x[0](newx, self.mhatt_x[0](
            newx, bert_x_k, bert_x_k, text_mask.unsqueeze(1)))  # KGnewx KG子图
        newx = self.res4ffn_x[0](newx, self.ffn_x[0](newx))
        x = torch.cat((bert_x, newx), dim=2)
        x = self.hidden2tag(x)

        return self.crf.decode(x, mask=crf_mask)
