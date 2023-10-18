# -*- coding:utf-8 -*-
# @Time    :2023/9/9 20:43
# @Author  :ZZK
# @ File   :data-preprocessing.py
# Description:
import spary
import glob
import os

labels_B = ["B-PER", "B-ORG", "B-LOC"]
labels_I = ["I-PER", "I-ORG", "I-LOC"]


def dataPre():
    files_l = glob.glob("*_l.txt")
    # print(files_l)
    for file_l in files_l:
        file_s = file_l.replace("_l.txt", "_s.txt")
        file_k = file_l.replace("_l.txt", "_k.txt")
        if os.path.exists(file_k):
            continue
        with open(file_l, 'r') as fl:
            with open(file_s, 'r', encoding="utf-8") as fs:
                print(file_s)
                con = fl.readline().split("\t")
                con_l = fs.readline().split("\t")
                entities = []  # "[CLS].join()"
                entity = ""
                for i in range(len(con)):
                    if con[i] == "O":
                        if entity != "":
                            entities.append(entity)
                            entity = ""
                    else:
                        entity += (con_l[i] + " ")
                if entity != "":
                    entities.append(entity)
        fk_con = []
        for ent in entities:
            print(ent)
            ent_k = spary.sparqlw(ent)
            fk_con.append(ent_k)
        print(fk_con)
        with open(file_k, 'w', encoding="utf-8") as fk:
            fk.write("\t".join(fk_con))


if __name__ == '__main__':
    while True:
        try:
            dataPre()
            break
        except:
            continue
