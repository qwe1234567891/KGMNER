# -*- coding:utf-8 -*-
# @Time    :2023/6/10 10:25
# @Author  :ZZK
# @ File   :my_dataset.py
# Description:
from torch.utils.data import Dataset
from transformers import BertTokenizer
import os
import glob

from utils import *
from config import tag2idx, max_len, max_node


class MMNerDataset(Dataset):
    def __init__(self, textdir, imgdir="./image/twitter2017/image"):
        self.X_files = sorted(glob.glob(os.path.join(textdir, "*_s.txt")))  # 句子
        self.K_files = sorted(glob.glob(os.path.join(textdir, "*_k.txt")))  # 扩充的实体
        self.Y_files = sorted(glob.glob(os.path.join(textdir, "*_l.txt")))  # 标签
        self.P_files = sorted(glob.glob(os.path.join(textdir, "*_p.txt")))  # 图片id
        self._imgdir = imgdir
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    def __len__(self):
        return len(self.X_files)

    def construct_inter_matrix(self, word_num, pic_num=max_node):  # 句子长度*4
        mat = np.zeros((max_len, pic_num), dtype=np.float32)  # 128*4
        mat[:word_num, :pic_num] = 1.0  # 第一维取前word_num行，第二维取前4列赋1，其余为0
        return mat

    def __getitem__(self, idx):  # idx是一个索引，取值范围取决于__len__函数中返回值。
        with open(self.X_files[idx], "r", encoding="utf-8") as fr:  # 索引为idx的句子
            s = fr.readline().split("\t")  # 分好词的

        with open(self.K_files[idx], "r", encoding="utf-8") as fr:  #
            k1 = fr.readline().split("\t")  #
            con_k = " ".join(k1)
            k = con_k.split(" ")

        with open(self.Y_files[idx], "r", encoding="utf-8") as fr:  # 索引为idx的标签
            l = fr.readline().split("\t")

        with open(self.P_files[idx], "r", encoding="utf-8") as fr:  # 索引为idx的图片
            imgid = fr.readline()  # 62654
            picpaths = [os.path.join(self._imgdir, "{}/{}.jpg".format(imgid,imgid))]  # 对应数据集中四张图片

        ntokens = ["[CLS]"]
        ntokens_k = ["[CLS]"]
        for word_k in k:
            tokens_k = self.tokenizer._tokenize(word_k)
            ntokens_k.append(str(tokens_k))
        label_ids = [tag2idx["CLS"]]  # 11  标签id
        for word, label in zip(s, l):  # iterate every word重复每个单词
            tokens = self.tokenizer._tokenize(word)  # one word may be split into several tokens一个词可以分成几个标记
            ntokens.extend(tokens)  # 加上[CLS],[tokens]
            for i, _ in enumerate(tokens):  # i是tokens的索引
                label_ids.append(tag2idx[label] if i == 0 else tag2idx["X"])

        ntokens = ntokens[:max_len - 1]  # max_len:128
        ntokens_k = ntokens_k[:max_len - 1]
        ntokens_k.append("[SEP]")
        ntokens.append("[SEP]")
        label_ids = label_ids[:max_len - 1]
        label_ids.append(tag2idx["SEP"])
        matrix = self.construct_inter_matrix(len(label_ids), len(picpaths))  # 句子index总数，图片数量，构建一个初始化矩阵

        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)  # 转id表示
        input_k_ids = self.tokenizer.convert_tokens_to_ids(ntokens_k)
        mask = [1] * len(input_ids)
        mask_k = [1] * len(input_k_ids)
        segment_ids = [0] * max_len

        pad_len = max_len - len(input_ids)
        pad_k_len = max_len - len(input_k_ids)
        rest_pad_k = [0] * pad_k_len
        input_k_ids.extend(rest_pad_k)  # importance
        rest_pad = [0] * pad_len  # pad to max_len
        input_ids.extend(rest_pad)  # 补0
        mask.extend(rest_pad)  # 补0
        mask_k.extend(rest_pad_k)
        label_ids.extend(rest_pad)  # 补0

        # pad ntokens  extend():在list中追加另一个list的所有值
        ntokens.extend(["pad"] * pad_len)
        ntokens_k.extend(["pad"] * pad_k_len)
        # print("input_ids:",input_ids)
        # print("input_k_ids:",input_k_ids)
        return {
            "ntokens": ntokens,
            "input_ids": input_ids,
            "segment_ids": segment_ids,
            "mask": mask,
            "mask_k": mask_k,
            "label_ids": label_ids,
            "picpaths": picpaths,
            "matrix": matrix,
            "ntokens_k": ntokens_k,
            "input_k_ids": input_k_ids
        }
class KGMMNerDataset(Dataset):
    def __init__(self, textdir, imgdir="./image/kgmner"):
        self.X_files = sorted(glob.glob(os.path.join(textdir, "*_s.txt")))  # 句子
        self.K_files = sorted(glob.glob(os.path.join(textdir, "*_k.txt")))  # 扩充的实体
        self.Y_files = sorted(glob.glob(os.path.join(textdir, "*_l.txt")))  # 标签
        self.P_files = sorted(glob.glob(os.path.join(textdir, "*_p.txt")))  # 图片id
        self._imgdir = imgdir
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    def __len__(self):
        return len(self.X_files)

    def construct_inter_matrix(self, word_num, pic_num=max_node):  # 句子长度*4
        mat = np.zeros((max_len, pic_num), dtype=np.float32)  # 128*4
        mat[:word_num, :pic_num] = 1.0
        return mat

    def __getitem__(self, idx):  # idx是一个索引，取值范围取决于__len__函数中返回值。
        with open(self.X_files[idx], "r", encoding="utf-8") as fr:  # 索引为idx的句子
            s = fr.readline().split("\t")  #

        with open(self.K_files[idx], "r", encoding="utf-8") as fr:  #
            k1 = fr.readline().split("\t")  #
            con_k = " ".join(k1)
            k = con_k.split(" ")

        with open(self.Y_files[idx], "r", encoding="utf-8") as fr:  # 索引为idx的标签
            l = fr.readline().split("\t")

        with open(self.P_files[idx], "r", encoding="utf-8") as fr:  # 索引为idx的图片
            imgid = fr.readline()  # 62654
            if imgid:  # 有的可能没有图片
                picpaths = [os.path.join(self._imgdir, "{}.jpg".format(imgid))]  # 对应的图片

        ntokens = ["[CLS]"]
        ntokens_k = ["[CLS]"]
        for word_k in k:
            ntokens_k.append(str(word_k))
        label_ids = [tag2idx["CLS"]]  # 标签id
        for word, label in zip(s, l):  # iterate every word重复每个单词
            ntokens.extend(word)  # 加上[CLS],[tokens]
            label_ids.append(tag2idx[label])

        ntokens = ntokens[:max_len - 1]  # max_len:128
        ntokens_k = ntokens_k[:max_len - 1]
        ntokens_k.append("[SEP]")
        ntokens.append("[SEP]")
        label_ids = label_ids[:max_len - 1]
        label_ids.append(tag2idx["SEP"])
        matrix = self.construct_inter_matrix(len(label_ids), len(picpaths))  # 句子index总数，图片数量，构建一个初始化矩阵

        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)  # 转id表示
        input_k_ids = self.tokenizer.convert_tokens_to_ids(ntokens_k)
        mask = [1] * len(input_ids)
        mask_k = [1] * len(input_k_ids)
        segment_ids = [0] * max_len

        pad_len = max_len - len(input_ids)
        pad_k_len = max_len - len(input_k_ids)
        rest_pad_k = [0] * pad_k_len
        input_k_ids.extend(rest_pad_k)  # importance
        rest_pad = [0] * pad_len  # pad to max_len
        input_ids.extend(rest_pad)  # 补0
        mask.extend(rest_pad)  # 补0
        mask_k.extend(rest_pad_k)
        label_ids.extend(rest_pad)  # 补0

        # pad ntokens  extend():在list中追加另一个list的所有值
        ntokens.extend(["pad"] * pad_len)
        ntokens_k.extend(["pad"] * pad_k_len)
        # print("input_ids:",input_ids)
        # print("input_k_ids:",input_k_ids)
        return {
            "ntokens": ntokens,
            "input_ids": input_ids,
            "segment_ids": segment_ids,
            "mask": mask,
            "mask_k": mask_k,
            "label_ids": label_ids,
            "picpaths": picpaths,
            "matrix": matrix,
            "ntokens_k": ntokens_k,
            "input_k_ids": input_k_ids
        }
