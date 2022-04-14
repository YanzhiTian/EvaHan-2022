import torch
from Deeper_Model import Deep_Model
from transformers import AutoTokenizer
import os
import eval_score


tokenizer = AutoTokenizer.from_pretrained('../sikubert')
idx2label = {
    0: '[PAD]', 1: '[CLS]',
    2: '[SEP]',
    3: 'b-a', 4: 'i-a',
    5: 'b-c', 6: 'i-c',
    7: 'b-d', 8: 'i-d',
    9: 'b-f', 10: 'i-f',
    11: 'b-j', 12: 'i-j',
    13: 'b-m', 14: 'i-m',
    15: 'b-n', 16: 'i-n',
    17: 'b-nr', 18: 'i-nr',
    19: 'b-ns', 20: 'i-ns',
    21: 'b-p', 22: 'i-p',
    23: 'b-q', 24: 'i-q',
    25: 'b-r', 26: 'i-r',
    27: 'b-s', 28: 'i-s',
    29: 'b-t', 30: 'i-t',
    31: 'b-u', 32: 'i-u',
    33: 'b-v', 34: 'i-v',
    35: 'b-sv', 36: 'i-sv',
    37: 'b-wv', 38: 'i-wv',
    39: 'b-yv', 40: 'i-yv',
    41: 'b-y', 42: 'i-y',
    43: 'b-w', 44: 'i-w',
    45: 'b-x', 46: 'i-x'
}


def save_result(file_path, sentence, idx_label):
    real_label = []
    sentence = sentence.replace('\n', '')
    for i in range(len(idx_label)):
        each_label = idx2label[idx_label[i]]
        if i == 0 or i == len(idx_label) - 1:  # 去除[CLS] [SEP]
            continue
        real_label.append(each_label)
    with open(file_path, 'a', encoding='utf-8') as f:
        for i in range(len(real_label)):
            if i+1 == len(real_label):
                f.write(sentence[i]+'/'+pos_i1+' ')
            else:
                pair_i = real_label[i].split('-')
                if pair_i == ['[CLS]'] or pair_i == ['[SEP]'] or pair_i == ['[PAD]']:  # 有时句末标点会被标为这几个
                    seg_i = 'b'
                    pos_i = 'w'
                else:
                    seg_i = pair_i[0]
                    pos_i = pair_i[1]
                f.write(sentence[i])
                pair_i1 = real_label[i+1].split('-')
                if pair_i1 == ['[CLS]'] or pair_i1 == ['[SEP]'] or pair_i1 == ['[PAD]']:
                    seg_i1 = 'b'
                    pos_i1 = 'w'
                else:
                    seg_i1 = pair_i1[0]
                    pos_i1 = pair_i1[1]
                if seg_i1 == 'b':
                    f.write('/'+pos_i+' ')
    f.close()


def check_unreadable_word(old_label, sentence):
    new_label = [1]  # [CLS]
    j = 1  # 新label的下标
    for i in range(len(sentence)):
        if len(tokenizer(sentence[i])['input_ids']) <= 2:
            new_label.append(45)  # b-x
        else:
            new_label.append(old_label[j])
            j += 1
    new_label.append(2)  # [SEP]
    return new_label


def dev_test():
    model = torch.load('bert_lstm_crf.model')
    model.eval()
    path = './data/predict.txt'
    if os.path.exists(path):
        os.remove(path)
    with open('./data/dev_only_text.txt', 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            line = line.replace('\n', '')
            if len(line) > 450:  # 防止超出512，在合理位置切断
                total_str = []
                temp_str = ''
                for each_word in line:
                    temp_str += each_word
                    if len(temp_str) > 450 and (each_word == '，' or each_word == '。'):
                        total_str.append(temp_str)
                        temp_str = ''
                total_str.append(temp_str)
                # print(total_str)
                for each_str in total_str:
                    label = model(each_str)[0]
                    if len(label) != len(each_str) + 2:
                        label = check_unreadable_word(label, each_str)
                    save_result('./data/predict.txt', each_str, label)
            else:
                label = model(line)[0]
                if len(label) != len(line) + 2:
                    label = check_unreadable_word(label, line)
                # print(line)
                save_result('./data/predict.txt', line, label)

            with open('./data/predict.txt', 'a', encoding='utf-8') as f1:
                f1.write('\n')  # 完全写完一句后，加回车
            f1.close()
            line = f.readline()
    f.close()
    eval_score.print_result()


if __name__ == "__main__":
    dev_test()
