import torch

"""
文本序列化
将文本转化成对应的张量才能进行处理
"""
class WordSequence:
    UNK_TAG = '<UNK>'
    PAD_TAG = '<PAD'
    # unk用来标记词典中未出现过的字符串
    # pad用来对不到设置的规定长度句子进行数字填充
    UNK = 0
    PAD = 1

    def __init__(self):
        # self.dict用来对于词典中每种给一个对应的序号
        self.dict = {
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }
        # 统计每种单词的数量
        self.count = {}

    def fit(self, sentence):
        """
        统计词频
        :param sentence: 一个句子 ['今','天','我','们','很','开','心']
        :return:
        """
        for word in sentence:
            # 字典(Dictionary) get(key,default=None) 函数返回指定键的值，如果值不在字典中返回默认值
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self, min_count=0, max_count=None, max_features=None):
        """
        根据条件构建词典
        :param min_count: 最小词频
        :param max_count: 最大词频
        :param max_features: 最大词语数
        :return:
        """
        if min_count is not None:
            # items()函数以列表返回可遍历的（键，值）元组数组
            self.count = {word: count for word, count in self.count.items() if count >= min_count}
        if max_count is not None:
            self.count = {word: count for word, count in self.count.items() if count <= max_count}
        if max_features is not None:
            # 排序
            self.count = dict(sorted(self.count.items(), key=lambda x: x[-1], reverse=True)[:max_features])
        for word in self.count:
            self.dict[word] = len(self.dict)

        # 把dict进行反转，就是键值和关键字进行反转
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence, max_len=None):
        """
        把句子转化为数字序列
        :param sentence: 句子
        :param max_len: 句子最大长度
        :return:
        """
        if len(sentence) > max_len:
            # 句子太长时进行截断
            sentence = sentence[:max_len]
        else:
            # 句子长度不够标准长度时，进行填充
            sentence = sentence + [self.PAD_TAG] * (max_len - len(sentence))
        # 句子中的单词没有出现在词典中设置为数字0（self.UNK)
        return [self.dict.get(word, self.UNK) for word in sentence]

    def inverse_transform(self, indices):
        """
        把数字序列转化为字符
        :param indices: 数字序列
        :return:
        """
        return [self.inverse_dict.get(i, '<UNK>') for i in indices]

    def __len__(self):
        # 返回词典个数
        return len(self.dict)

def load_data_to_model():
    """
    将原始数据序列化到pkl模型
    :return:
    """
    ws = WordSequence()
    data_path = r"data/aclImdb"
    total_path = []
    for temp_path in [r"/train/pos", r"/test/neg"]:
        cur_path = data_path + temp_path
        # 添加积极和消极评论的所有文件
        total_path += [os.path.join(cur_path, i) for i in os.listdir(cur_path) if i.endswith(".txt")]
    for file in tqdm(total_path, total=len(total_path)):
        # 读取评论内容
        content = open(file=file, encoding='utf-8').read()
        # 将评论分成一个个单词列表
        sentence = tokenize(content)  # tokenize有问题
        ws.fit(sentence)
    # 开始构建词典
    ws.build_vocab(min_count=5, max_count=15000)
    print(len(ws))
    # dump将数据通过特殊的形式转换为只有python语言认识的字符串，并写入文件
    pickle.dump(ws, open("./model/ws.pkl", "wb"))



import pickle
import torch

# load 从数据文件中读取数据，并转换为python的数据结构
ws = pickle.load(open("./model/ws.pkl", "rb"))
embedding_dim = 256
hidden_size = 128
num_layers = 2
dropout = 0.5
train_batch_size = 1000
test_batch_size = 1000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import re


def tokenize(sentence):
    """
    函数功能：将一个句子拆分成一个个单词列表
    1、先说sub是替换的意思。
    2、.是匹配任意字符（除换行符外）*是匹配前面的任意字符一个或多个
    3、？是非贪婪。
    4、组合起来的意思是将"<"和中间的任意字符">" 换为空字符串""
    由于有？是非贪婪。 所以是匹配"<"后面最近的一个">"
    :param sentence:
    :return:
    """
    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                '\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”',
                '“', ]
    sentence = sentence.lower()  # 把大写转化为小写
    sentence = re.sub("<br />", " ", sentence)
    # sentence = re.sub("I'm","I am",sentence)
    # sentence = re.sub("isn't","is not",sentence)
    sentence = re.sub("|".join(fileters), " ", sentence)
    result = [i for i in sentence.split(" ") if len(i) > 0]

    return result


from torch.utils.data import Dataset, DataLoader
import random
import os


class ImdbDataset(Dataset):
    def __init__(self, model='train'):  #train,val,test
        super(ImdbDataset, self).__init__()
        self.model=model
        data_path = r"data/aclImdb"
        if self.model == 'train' or self.model=='val':
            data_path+= r"/train"
        elif self.model == 'test':
             data_path+= r"/train"
        self.total_path = []
        for temp_path in [r"/pos", r"/neg"]:
            cur_path = data_path + temp_path
            # 添加积极和消极评论的所有文件
            self.total_path += [os.path.join(cur_path, i) for i in os.listdir(cur_path) if i.endswith(".txt")]
        if self.model == 'train' or self.model=='val':
            ration=0.95  #trian和val比例
            train_num=int(round(ration*len(self.total_path)))
            if self.model == 'train':
                self.total_path=random.sample(self.total_path,train_num)
            else:
                self.total_path=random.sample(self.total_path,len(self.total_path)-train_num)

    def __getitem__(self, idx):
        # 获取某评论的文件路径
        file = self.total_path[idx]
        # 读取评论内容
        content = open(file=file, encoding='utf-8').read()
        # 将评论分成一个个单词列表
        content = tokenize(content)
        # 获取评论的分数（小于5为消极，大于等于5为积极）
        score = int(file.split("_")[1].split(".")[0])
        label = 0 if score < 5 else 1
        return content, label

    def __len__(self):
        return len(self.total_path)


def collate_fn(batch):
    """
    对batch数据进行处理([tokens,label],[tokens,label]...)
    :param batch:
    :return:
    """
    # *batch 可理解为解压，返回二维矩阵式
    content, labels = list(zip(*batch))
    # content中是有batch_size个评论（句子）


    content = [ws.transform(sentence, 200) for sentence in content]
    # content式字符串数组，必须先将数组中字符转化成对应数字，才能转成张量
    content = torch.LongTensor(content)
    labels = torch.LongTensor(labels)

    return content, labels


def get_dataloader(model='train'):
    imdb_dataset = ImdbDataset(model)
    return DataLoader(imdb_dataset,
                      batch_size=train_batch_size if model=='train' else test_batch_size,
                      shuffle=True if model=='train' else False,
                      collate_fn=collate_fn)


"""
定义模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_Model(nn.Module):
    def __init__(self):
        super(LSTM_Model, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(ws),  # 词典的大小
                                      embedding_dim=embedding_dim)  # 256
        self.lstm = nn.LSTM(input_size=embedding_dim,  # 256
                            hidden_size=hidden_size,  # 128
                            num_layers=num_layers,  # 2
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout)  # 0.5
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # [128*2,128]
        self.fc2 = nn.Linear(hidden_size, 2)  # [128,2]

    def forward(self, input):
        """
        :param input: [batch_size,max_len],其中max_len表示每个句子有多少单词
        :return:
        """
        x = self.embedding(input)  # [batch_size,max_len,embedding_dim]
        # 经过lstm层，x:[batch_size,max_len,2*hidden_size]
        # h_n,c_n:[2*num_layers,batch_size,hidden_size]
        out, (h_n, c_n) = self.lstm(x)

        # 获取两个方向最后一次的h，进行concat
        output_fw = h_n[-2, :, :]  # [batch_size,hidden_size]
        output_bw = h_n[-1, :, :]  # [batch_size,hidden_size]
        out_put = torch.cat([output_fw, output_bw], dim=-1)  # [batch_size,hidden_size*2]

        out_fc1 = F.relu(self.fc1(out_put))  # []
        out_put = self.fc2(out_fc1)
        return F.log_softmax(out_put, dim=-1)


import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt



loss_list = []


def train(epoch,model,optimizer,train_dataloader):
    model.train()
    bar = tqdm(train_dataloader, total=len(train_dataloader))  #配置进度条
    for idx, (input, target) in enumerate(bar):
        input = input.to(device)
        target = target.to(device)
        output = model(input)
        loss = F.nll_loss(output, target)
        optimizer.zero_grad()
        loss.backward()
        #loss_list.append(loss.cpu().data)
        #print(loss.cpu ().item())
        loss_list.append (loss.cpu().item())
        optimizer.step()
        bar.set_description("epoch:{} idx:{} loss:{:.6f}".format(epoch, idx, np.mean(loss_list)))

def eval(model,test_dataloader):
    model.eval()
    loss_list = []
    eval_acc=0
    eval_total=0
    with torch.no_grad():
        for input, target in test_dataloader:
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            loss = F.nll_loss(output, target)
            loss_list.append(loss.item())
            # 准确率
            output_max = output.max(dim=-1) #返回最大值和对应的index
            pred = output_max[-1]  #最大值的index
            eval_acc+=pred.eq(target).cpu().float().sum().item()
            eval_total+=target.shape[0]
        acc=eval_acc/eval_total
        print("loss:{:.6f},acc:{}".format(np.mean(loss_list), acc))
    return acc

# 如果训练的时间足够长，可以达到99%以上的准确度

def test(test_dataloader):
    model = LSTM_Model().to(device)
    model.load_state_dict(torch.load('model/model.pkl'))
    model.eval()
    loss_list = []
    test_acc=0
    test_total=0
    bar=tqdm(test_dataloader,total=len(test_dataloader))
    with torch.no_grad():
        for input, target in bar:
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            loss = F.nll_loss(output, target)
            loss_list.append(loss.item())
            # 准确率
            output_max = output.max(dim=-1) #返回最大值和对应的index
            pred = output_max[-1]  #最大值的index
            test_acc+=pred.eq(target).cpu().float().sum().item()
            test_total+=target.shape[0]
        print("test loss:{:.6f},acc:{}".format(np.mean(loss_list), test_acc/test_total))

if __name__ == '__main__':
    model = LSTM_Model().to(device)
    count_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {count_parameters:,} trainable parameters')
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_dataloader = get_dataloader(model='train')
    val_dataloader = get_dataloader(model='val')
    best_acc=0
    early_stop_cnt=0

    for epoch in range(5):
        load_data_to_model()
        train(epoch,model,optimizer,train_dataloader)
        acc=eval(model,val_dataloader)
        if acc>best_acc:
            best_acc=acc
            torch.save(model.state_dict(), 'model/model.pkl')
            torch.save(optimizer.state_dict(), 'model/optimizer.pkl')
            print("save model,acc:{}".format(best_acc))
            early_stop_cnt=0
        else:
            early_stop_cnt+=1
        if early_stop_cnt>5:
            break
    plt.figure(figsize=(20, 8))
    plt.plot(range(len(loss_list)), loss_list)

    test_dataloader=get_dataloader(model='test')
    test(test_dataloader)