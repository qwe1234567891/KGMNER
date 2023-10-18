# -*- encoding: utf-8 -*-
'''
@File     : kgmner.py
@DateTime : 2023/6/9 00:15:41
@Author   : zzk
@Desc     : 
'''
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms
from PIL import Image
import os
import datetime
import time
import random
import argparse
from tqdm import tqdm
import warnings

from utils import *
from metric import evaluate_pred_file
from config import tag2idx, idx2tag, max_node, log_fre
from my_dataset import MMNerDataset
from my_model import MMNerModel

warnings.filterwarnings("ignore")
predict_file = "./output/twitter2017/{}/epoch_{}.txt"
device = torch.device("cuda:0")
# device = torch.device("cpu")
if not os.path.exists("./output/twitter2015/val"):
    os.makedirs("./output/twitter2015/val")
if not os.path.exists("./output/twitter2015/test"):
    os.makedirs("./output/twitter2015/test")

if not os.path.exists("./output/twitter2017/val"):
    os.makedirs("./output/twitter2017/val")
if not os.path.exists("./output/twitter2017/test"):
    os.makedirs("./output/twitter2017/test")


def seed_torch(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def cos(pic1, pic2):
    resnet = models.resnet50(pretrained=True)
    modules = list(resnet.children())[:-1]
    resnet = nn.Sequential(*modules)
    resnet.eval()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    img1 = Image.open(pic1)
    img2 = Image.open(pic2)
    img01 = transform(img1)
    img02 = transform(img2)
    features1 = resnet(img01.unsqueeze(0))
    features2 = resnet(img02.unsqueeze(0))
    return torch.cosine_similarity(features1, features2).item()


def collate_fn(batch):
    input_ids = []
    input_k_ids = []
    token_type_ids = []
    attention_mask = []
    attention_mask_k = []
    label_ids = []

    b_ntokens = []
    b_matrix = []
    b_img = torch.zeros(len(batch) * max_node, 3, 224, 224)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    for idx, example in enumerate(batch):
        b_ntokens.append(example["ntokens"])
        input_ids.append(example["input_ids"])
        input_k_ids.append(example["input_k_ids"])
        token_type_ids.append(example["segment_ids"])
        attention_mask.append(example["mask"])
        attention_mask_k.append(example["mask_k"])
        label_ids.append(example["label_ids"])
        b_matrix.append(example["matrix"])

        for i, picpath in enumerate(example["picpaths"]):
            try:

                b_img[idx * max_node + i] = preprocess(Image.open(picpath).convert('RGB'))
            except:
                print("========={} error!===============".format(picpath))
                exit(1)

    return {
        "b_ntokens": b_ntokens,
        "x": {
            "input_ids": torch.tensor(input_ids).to(device),
            "token_type_ids": torch.tensor(token_type_ids).to(device),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.uint8).to(device),
            "input_k_ids": torch.tensor(input_k_ids).to(device),
            "attention_mask_k": torch.tensor(attention_mask_k, dtype=torch.uint8).to(device)
        },
        "b_img": torch.tensor(b_img).to(device),
        "b_matrix": torch.tensor(b_matrix).to(device),
        "y": torch.tensor(label_ids).to(device)
    }


def save_model(model, model_path="./model.pt"):
    torch.save(model.state_dict(), model_path)
    print("Current Best mmner model has beed saved!")


def predict(epoch, model, dataloader, mode="val", res=None):
    '''
    res["best_f1"] = 0.0   res["epoch"] = -1
    '''
    model.eval()
    with torch.no_grad():
        filepath = predict_file.format(mode, epoch)  # /output/twitter2015/val/epoch_n.txt
        with open(filepath, "w", encoding="utf8") as fw:
            for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Predicting"):  # 进度条
                b_ntokens = batch["b_ntokens"]

                x = batch["x"]
                b_img = batch["b_img"]
                inter_matrix = batch["b_matrix"]
                text_mask = x["attention_mask"]
                y = batch["y"]
                output = model(x, b_img, inter_matrix, text_mask, y)

                # write into file
                for idx, pre_seq in enumerate(output):
                    ground_seq = y[idx]
                    for pos, (pre_idx, ground_idx) in enumerate(zip(pre_seq, ground_seq)):
                        if ground_idx == tag2idx["PAD"] or ground_idx == tag2idx["X"] or ground_idx == tag2idx[
                            "CLS"] or ground_idx == tag2idx["SEP"]:
                            continue
                        else:
                            predict_tag = idx2tag[pre_idx] if idx2tag[pre_idx] not in [
                                "PAD", "X", "CLS", "SEP"] else "O"
                            true_tag = idx2tag[ground_idx.data.item()]
                            line = "{}\t{}\t{}\n".format(b_ntokens[idx][pos], predict_tag, true_tag)
                            fw.write(line)
        print("=============={} -> {} epoch done=================".format(mode, epoch))
        cur_f1 = evaluate_pred_file(filepath)
        to_save = False
        if mode == "test":
            if res["best_f1"] < cur_f1:
                res["best_f1"] = cur_f1
                res["epoch"] = epoch
                to_save = True
            print("current best f1: {}, epoch: {}".format(res["best_f1"], res["epoch"]))
        return to_save


def train(args):
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    seed_torch(args.seed)

    train_textdir = os.path.join(args.txtdir, "train")
    train_dataset = MMNerDataset(textdir=train_textdir, imgdir=args.imgdir)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=collate_fn)

    val_textdir = os.path.join(args.txtdir, "valid")
    val_dataset = MMNerDataset(textdir=val_textdir, imgdir=args.imgdir)
    val_dataloader = DataLoader(val_dataset, batch_size=args.train_batch_size, collate_fn=collate_fn)

    test_textdir = os.path.join(args.txtdir, "test")
    test_dataset = MMNerDataset(textdir=test_textdir, imgdir=args.imgdir)
    test_dataloader = DataLoader(test_dataset, batch_size=args.train_batch_size, collate_fn=collate_fn)

    model = MMNerModel()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.8)

    res = {}
    res["best_f1"] = 0.0
    res["epoch"] = -1
    start = time.time()
    for epoch in range(args.num_train_epoch):
        model.train()
        print(len(train_dataloader))
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            x = batch["x"]
            b_img = batch["b_img"]
            inter_matrix = batch["b_matrix"]
            text_mask = x["attention_mask"]
            y = batch["y"]
            loss = model.likelihood(x, b_img, inter_matrix, text_mask, y)  # log_likelihood最大似然函数
            loss.backward()
            optimizer.step()

            if i % log_fre == 0:
                print("EPOCH: {} Step: {} Loss: {}".format(epoch, i, loss.data))

        scheduler.step()
        predict(epoch, model, val_dataloader, mode="val", res=res)
        to_save = predict(epoch, model, test_dataloader, mode="test", res=res)
        if to_save:
            save_model(model, args.ckpt_path)

    print("================== train done! ================")
    end = time.time()
    hour = int((end - start) // 3600)
    minute = int((end - start) % 3600 // 60)
    now = datetime.datetime.now()
    print("total time: {} hour - {} minute".format(hour, minute), now.strftime("%Y-%m-%d %H:%M:%S"))


def test(args):
    model = MMNerModel().to(device)
    model.load_state_dict(torch.load(args.ckpt_path))

    test_textdir = os.path.join(args.txtdir, "test")
    test_dataset = MMNerDataset(textdir=test_textdir, imgdir=args.imgdir)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, collate_fn=collate_fn)

    model.eval()
    with torch.no_grad():
        filepath = "./test_output.txt"
        with open(filepath, "w", encoding="utf8") as fw:
            for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Testing"):
                b_ntokens = batch["b_ntokens"]
                x = batch["x"]
                b_img = batch["b_img"]
                inter_matrix = batch["b_matrix"]
                text_mask = x["attention_mask"]
                y = batch["y"]
                output = model(x, b_img, inter_matrix, text_mask, y)

                for idx, pre_seq in enumerate(output):
                    ground_seq = y[idx]
                    for pos, (pre_idx, ground_idx) in enumerate(zip(pre_seq, ground_seq)):
                        if ground_idx == tag2idx["PAD"] or ground_idx == tag2idx["X"] or ground_idx == tag2idx[
                            "CLS"] or ground_idx == tag2idx["SEP"]:
                            continue
                        else:
                            predict_tag = idx2tag[pre_idx] if idx2tag[pre_idx] not in [
                                "PAD", "X", "CLS", "SEP"] else "O"
                            true_tag = idx2tag[ground_idx.data.item()]
                            line = "{}\t{}\t{}\n".format(b_ntokens[idx][pos], predict_tag, true_tag)
                            fw.write(line)
        evaluate_pred_file(filepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run training.")
    parser.add_argument('--txtdir', type=str, default="./text/twitter2015/", help="text dir")
    parser.add_argument('--imgdir', type=str, default="./image/twitter2015/image/", help="image dir")
    parser.add_argument('--ckpt_path', type=str, default="./model.pt", help="path to save or load model")
    parser.add_argument("--num_train_epoch", default=30, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Total batch size for training.")
    parser.add_argument("--test_batch_size", default=16, type=int, help="Total batch size for eval.")
    parser.add_argument("--lr", default=0.0001, type=float, help="The initial learning rate for Adam.")
    parser.add_argument('--seed', type=int, default=2021, help="random seed for initialization")
    args = parser.parse_args()
    if args.do_train:
        train(args)
    elif args.do_test:
        test(args)
    else:
        raise ValueError('At least one of `do_train`, `do_eval` must be True.')


if __name__ == "__main__":
    main()
