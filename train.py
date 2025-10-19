import os
import random
import time

import torch
import numpy as np
import torch.nn as nn
import math

# from models.model import Model
from models.model import Model
from utils import load_adj_cooccur, load_adj_temporal, EHRDataset, format_time, MultiStepLRScheduler
from metrics import evaluate_codes, evaluate_hf


def historical_hot(code_x, code_num, lens):
    result = np.zeros((len(code_x), code_num), dtype=int)  #
    for i, (x, l) in enumerate(zip(code_x, lens)):
        result[i] = x[l - 1]
    return result


class SingleHeadAttentionLayer1(nn.Module):
    def __init__(self, query_size, key_size, value_size, attention_size):
        super().__init__()
        self.attention_size = attention_size
        self.dense_q = nn.Linear(query_size, attention_size)
        self.dense_k = nn.Linear(key_size, attention_size)
        self.dense_v = nn.Linear(value_size, value_size)

    def forward(self, q, k, v):
        # 获取输入张量的设备
        device = q.device
        # 确保模型参数在正确的设备上
        self.dense_q = self.dense_q.to(device)
        self.dense_k = self.dense_k.to(device)
        self.dense_v = self.dense_v.to(device)

        query = self.dense_q(q.float())
        key = self.dense_k(k.float())
        value = self.dense_v(v.float())

        g = torch.div(torch.matmul(query, key.T), math.sqrt(self.attention_size))
        score = torch.softmax(g, dim=-1)  # 每个故障的权重
        output = torch.sum(torch.unsqueeze(score, dim=-1) * value, dim=-2)
        return output


if __name__ == '__main__':
    seed = 4669
    dataset = 'vehicle'  # 'mimic3' or 'eicu'
    task = 'm'  # 'm' or 'h'
    use_cuda = True
    if torch.cuda.is_available():
        device = torch.device('cuda:1')  # Specify GPU 1 (RTX 3090)
    else:
        device = 'cpu'
    code_size = 2613
    graph_size = 32
    hidden_size = 200  # rnn hidden size
    t_attention_size = 32
    t_output_size = hidden_size
    batch_size = 32
    epochs = 100
    num_heads = 2

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset_path = os.path.join('data', dataset, 'standard')
    train_path = os.path.join(dataset_path, 'train')
    valid_path = os.path.join(dataset_path, 'valid')
    test_path = os.path.join(dataset_path, 'test')

    adj_cooccur = load_adj_cooccur(dataset_path, device=device)
    adj_temporal = load_adj_temporal(dataset_path, device=device)
    code_num = len(adj_cooccur)
    print('loading train data ...')
    train_data = EHRDataset(train_path, label=task, batch_size=batch_size, shuffle=True, device=device)
    print('loading valid data ...')
    valid_data = EHRDataset(valid_path, label=task, batch_size=batch_size, shuffle=False, device=device)

    test_historical = historical_hot(valid_data.code_x, code_num, valid_data.visit_lens)

    task_conf = {
        'm': {
            'dropout': 0.5,
            'output_size': code_num,
            'evaluate_fn': evaluate_codes,
            'lr': {
                'init_lr': 0.001,
                'milestones': [10, 20],
                'lrs': [1e-3, 1e-5]
            }
        },
        'h': {
            'dropout': 0.0,
            'output_size': 1,
            'evaluate_fn': evaluate_hf,
            'lr': {
                'init_lr': 0.01,
                'milestones': [2, 3, 20],
                'lrs': [1e-3, 1e-4, 1e-5]
            }
        }
    }
    output_size = task_conf[task]['output_size']  # 输出的维度
    activation = torch.nn.Sigmoid()  # 激活函数
    loss_fn = torch.nn.BCELoss()  # 损失函数
    evaluate_fn = task_conf[task]['evaluate_fn']  # 评估函数
    dropout_rate = task_conf[task]['dropout']  # 丢包率

    param_path = os.path.join('data', 'params', dataset, task)
    if not os.path.exists(param_path):
        os.makedirs(param_path)

    model = Model(code_num=code_num, code_size=code_size,
                  adj=adj_cooccur, graph_size=graph_size, hidden_size=hidden_size, t_attention_size=t_attention_size,
                  t_output_size=t_output_size,
                  output_size=output_size, dropout_rate=dropout_rate, activation=activation).to(device)
    model = model.to(torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = MultiStepLRScheduler(optimizer, epochs, task_conf[task]['lr']['init_lr'],
    #                                  task_conf[task]['lr']['milestones'], task_conf[task]['lr']['lrs'])
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    for epoch in range(epochs):
        print('Epoch %d / %d:' % (epoch + 1, epochs))
        model.train()  # 开始训练
        total_loss = 0.0
        total_num = 0
        steps = len(train_data)
        st = time.time()
        # scheduler.step() # 标志开始优化
        for step in range(len(train_data)):
            optimizer.zero_grad()  # 优化置0
            code_x, visit_lens, y, neighbors = train_data[step]
            adj_cooccur = adj_cooccur.to(torch.float32)
            code_x = code_x.to(torch.float32)
            output1 = model(adj_cooccur, code_x, visit_lens)
            output = output1.squeeze()  # .squeeze()  降维升维
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * output_size * len(code_x)
            total_num += len(code_x)
            end_time = time.time()
            remaining_time = format_time((end_time - st) / (step + 1) * (steps - step - 1))
            print('\r    Step %d / %d, remaining time: %s, loss: %.4f'
                  % (step + 1, steps, remaining_time, total_loss / total_num), end='')
        train_data.on_epoch_end()
        et = time.time()
        time_cost = format_time(et - st)
        print('\r    Step %d / %d, time cost: %s, loss: %.4f' % (steps, steps, time_cost, total_loss / total_num))
        valid_loss, f1_score = evaluate_fn(adj_cooccur, adj_temporal, model, valid_data, loss_fn, output_size,
                                           test_historical)
