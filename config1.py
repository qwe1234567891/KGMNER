max_len = 128
max_node = 4
log_fre = 10
tag2idx = {
    "PAD": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-LOC": 3,
    "I-LOC": 4,
    "B-ORG": 5,
    "I-ORG": 6,
    "B-MEE": 7,
    "I-MEE": 8,
    "B-PRD": 9,
    "I-PRD": 10,
    "B-ACR": 11,
    "I-ACR": 12,
    "B-OTHER": 13,
    "I-OTHER": 14,
    "O": 15,
    "X": 16,
    "CLS": 17,
    "SEP": 18
}

idx2tag = {idx: tag for tag, idx in tag2idx.items()}
