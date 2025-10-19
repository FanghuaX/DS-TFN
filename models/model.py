import math
import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class SelfAttention(nn.Module):
    def __init__(self, size):
        super(SelfAttention, self).__init__()
        # 定义Q,K,V的变换矩阵，这里它们的维度都设置为150
        self.query = nn.Linear(size, size)
        self.key = nn.Linear(size, size)
        self.value = nn.Linear(size, size)

    def forward(self, x):
        # x is (batch_size, seq_len, size)
        # 对应 输入维度 32 x 69 x 150
        batch_size, seq_len, size = x.size()

        # 生成查询、键、值
        Q = self.query(x)  # (batch_size, seq_len, size)
        K = self.key(x)  # (batch_size, seq_len, size)
        V = self.value(x)  # (batch_size, seq_len, size)

        # 计算查询和键的点积,除以size的平方根进行缩放
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / size ** 0.5  # (batch_size, seq_len, seq_len)

        # 应用softmax函数得到的是每个序列的权重,注意mask
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, seq_len, seq_len)

        # 计算加权和，得到注意力后的输出
        output = torch.matmul(attention_weights, V)  # (batch_size, seq_len, size)

        return output


class Classifier(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0., activation=None):
        super().__init__()
        self.linear = nn.Linear(2613, output_size)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        output = self.dropout(x)
        output = self.linear(output)
        if self.activation is not None:
            output = self.activation(output)
        return output


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, lens):
        """hidden_states: (batch, seq_len, hidden_size)"""
        # 能量计算
        energy = torch.tanh(self.W(hidden_states))  # (batch, seq_len, hidden_size)
        scores = self.v(energy).squeeze(2)  # (batch, seq_len)

        # 创建掩码
        max_len = scores.size(1)
        mask = torch.arange(max_len).expand(len(lens), max_len).to(lens.device) < lens.unsqueeze(1)

        # 应用掩码
        scores[~mask] = -1e10
        weights = F.softmax(scores, dim=1)  # (batch, seq_len)

        # 加权求和
        weighted = torch.sum(hidden_states * weights.unsqueeze(2), dim=1)  # (batch, hidden_size)
        return weighted


class EfficientGraphFusion(nn.Module):
    def __init__(self, num_codes, code_size, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_codes = num_codes
        self.hidden_size = hidden_size
        self.code_size = code_size
        self.num_heads = num_heads

        # 代码嵌入层 - 添加归一化
        self.code_embed = nn.Embedding(num_codes, hidden_size)
        self.code_norm = nn.LayerNorm(hidden_size)

        # 高效图注意力层
        self.gat_layers = nn.ModuleList([
            SparseGraphAttention(hidden_size, hidden_size // num_heads, dropout=dropout)
            for _ in range(num_heads)
        ])
        self.merge_heads = nn.Linear(hidden_size, hidden_size)
        self.merge_norm = nn.LayerNorm(hidden_size)
        self.merge_dropout = nn.Dropout(dropout)

        # 动态图构建 - 使用更高效的参数化方式
        self.graph_builder = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            # nn.LeakyReLU(0.2),
            # nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        # 可学习的静态-动态融合参数
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # 事件聚合 - 使用GRU替代LSTM，更轻量
        self.aggregator = nn.GRU(hidden_size, hidden_size, batch_first=True)

        # 特殊标记用于无活跃代码的情况
        self.no_code_embed = nn.Parameter(torch.randn(1, hidden_size))

        # 静态图预处理
        self.static_proj = nn.Linear(1, 1)  # 用于调整静态邻接矩阵的权重

    def forward(self, time_sequence, static_adj):
        """
        time_sequence: (seq_len, code_size)
        static_adj: (num_codes, num_codes)
        返回: (hidden_size)
        """
        seq_len = time_sequence.size(0)
        device = time_sequence.device

        # 1. 获取代码嵌入并归一化
        code_embeds = self.code_norm(self.code_embed(torch.arange(self.num_codes).to(device)))

        # 2. 预计算静态图表示
        static_adj = self.static_proj(static_adj.unsqueeze(-1)).squeeze(-1)  # 调整静态权重

        # 3. 构建动态邻接矩阵
        dynamic_adj = self.build_dynamic_graph(code_embeds, static_adj)

        # 4. 处理每个时间步
        event_reps = []
        prev_event_rep = self.no_code_embed  # 初始使用特殊标记

        for t in range(seq_len):
            # 获取当前时间步的活跃代码
            active_mask = time_sequence[t] > 0
            active_indices = torch.where(active_mask)[0]

            if active_indices.numel() == 0:
                # 无活跃代码时使用前一个事件表示
                event_rep = prev_event_rep
            else:
                # 获取子图
                sub_embeds = code_embeds[active_indices]

                # 提取子图邻接矩阵
                sub_adj = dynamic_adj[active_indices][:, active_indices]

                # 多头图注意力
                head_outputs = []
                for gat_layer in self.gat_layers:
                    head_out = gat_layer(sub_embeds, sub_adj)
                    head_outputs.append(head_out)

                # 合并多头结果
                merged = torch.cat(head_outputs, dim=-1)
                aggregated = self.merge_heads(merged)
                aggregated = self.merge_norm(aggregated)
                aggregated = self.merge_dropout(aggregated)

                # 事件级表示 (平均池化)
                event_rep = torch.mean(aggregated, dim=0, keepdim=True)

            event_reps.append(event_rep)
            prev_event_rep = event_rep  # 更新前一个事件表示

        # 5. 时序聚合
        event_reps = torch.cat(event_reps, dim=0).unsqueeze(0)  # (1, seq_len, hidden_size)
        _, hidden = self.aggregator(event_reps)

        # 修正输出形状问题 - 确保输出为 (hidden_size)
        return hidden.squeeze(0)  # 移除第一个维度 (num_layers * num_directions, batch, hidden_size) -> (batch, hidden_size)

    def build_dynamic_graph(self, code_embeds, static_adj):
        # 只计算实际存在的边
        src, dst = torch.nonzero(static_adj > 0, as_tuple=True)  # 只处理静态图中存在的边

        if len(src) == 0:
            return static_adj  # 如果没有边，直接返回静态图

        # 计算动态权重
        src_emb = code_embeds[src]
        dst_emb = code_embeds[dst]
        pair_features = torch.cat([src_emb, dst_emb], dim=-1)
        dynamic_weights = torch.sigmoid(self.graph_builder(pair_features)).squeeze()

        # 创建动态邻接矩阵
        dynamic_adj = torch.zeros_like(static_adj)
        dynamic_adj[src, dst] = dynamic_weights
        dynamic_adj[dst, src] = dynamic_weights  # 假设图是无向的

        # 融合静态和动态邻接矩阵
        fused_adj = static_adj * (1 - self.alpha) + dynamic_adj * self.alpha

        return fused_adj

class SparseGraphAttention(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout

        # 权重矩阵
        self.W = nn.Linear(in_features, out_features, bias=False)

        # 注意力参数
        self.attn_src = nn.Parameter(torch.Tensor(1, out_features))
        self.attn_dst = nn.Parameter(torch.Tensor(1, out_features))
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, x, adj):
        """
        x: (N, in_features) 节点特征
        adj: (N, N) 稀疏邻接矩阵
        返回: (N, out_features) 更新后的节点特征
        """
        N = x.size(0)
        h = self.W(x)  # (N, out_features)

        # 高效计算注意力分数
        attn_src = (h * self.attn_src).sum(dim=-1)  # (N,)
        attn_dst = (h * self.attn_dst).sum(dim=-1)  # (N,)
        attn_scores = attn_src.unsqueeze(0) + attn_dst.unsqueeze(1)  # (N, N)

        # 应用邻接矩阵掩码
        mask = (adj > 0).float()
        attn_scores = self.leakyrelu(attn_scores) * mask

        # 避免数值溢出
        attn_scores = attn_scores - torch.max(attn_scores, dim=1, keepdim=True)[0]
        exp_scores = torch.exp(attn_scores) * mask

        # 计算归一化系数
        sum_exp = torch.sum(exp_scores, dim=1, keepdim=True) + 1e-8

        # 计算注意力权重
        attn_weights = exp_scores / sum_exp  # (N, N)
        attn_weights = F.dropout(attn_weights, self.dropout, training=self.training)

        # 聚合邻居信息
        output = torch.matmul(attn_weights, h)  # (N, out_features)
        return F.elu(output)


class EnhancedGatedTemporalFusion(nn.Module):
    def __init__(self, input_dim, global_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.global_dim = global_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # 确保 embed_dim 能够被 num_heads 整除
        assert global_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # 多尺度全局信息融合
        self.multihead_attn = nn.MultiheadAttention(global_dim, num_heads)

        # 动态位置编码
        self.dynamic_position_enc = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # 多层次门控机制
        self.gate1 = nn.Sequential(
            nn.Linear(input_dim + global_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.gate2 = nn.Sequential(
            nn.Linear(input_dim + global_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.candidate1 = nn.Sequential(
            nn.Linear(input_dim + global_dim, hidden_dim),
            nn.Tanh()
        )
        self.candidate2 = nn.Sequential(
            nn.Linear(input_dim + global_dim, hidden_dim),
            nn.Tanh()
        )

        # 输出投影
        self.output_proj = nn.Linear(hidden_dim,2613)
        self.output_proj1 = nn.Linear(100, 300)

    def forward(self, time_steps, global_code, positions):
        """
        time_steps: (seq_len, input_dim) - time-step representations
        global_code: (global_dim) - global sequence representation
        positions: (seq_len) - time positions
        """
        seq_len = time_steps.size(0)

        # 动态位置编码
        pos_emb = self.dynamic_position_enc(positions.float().unsqueeze(1))  # (seq_len, input_dim)
        time_steps = time_steps + pos_emb

        # 多尺度全局信息融合
        global_code = global_code.unsqueeze(0)  # (1, global_dim)
        global_code, _ = self.multihead_attn(global_code, global_code, global_code)
        global_code = global_code.squeeze(0)  # (global_dim)

        # 初始化隐藏状态
        hidden1 = torch.zeros(self.hidden_dim, device=time_steps.device)
        hidden2 = torch.zeros(self.hidden_dim, device=time_steps.device)
        outputs = []

        # 循环处理每个时间步
        for t in range(seq_len):
            # 拼接当前输入和全局信息
            # combined = torch.cat([time_steps[t], global_code], dim=-1)
            combined = time_steps[t]
            combined = self.output_proj1(combined)
            # 计算第一层门控值
            gate_val1 = self.gate1(combined)
            candidate_val1 = self.candidate1(combined)
            hidden1 = gate_val1 * hidden1 + (1 - gate_val1) * candidate_val1

            # 计算第二层门控值
            gate_val2 = self.gate2(combined)
            candidate_val2 = self.candidate2(combined)
            hidden2 = gate_val2 * hidden2 + (1 - gate_val2) * candidate_val2

            # 合并两层隐藏状态
            hidden = hidden1 + hidden2
            outputs.append(hidden)

        # 聚合所有隐藏状态
        outputs = torch.stack(outputs)  # (seq_len, hidden_dim)

        # 加权聚合 - 最近时间步权重更高
        weights = torch.softmax(torch.arange(seq_len, 0, -1, device=outputs.device).float(), dim=0)
        weighted_output = (outputs * weights.unsqueeze(1)).sum(dim=0)

        return self.output_proj(weighted_output)  # (input_dim)



class Model(nn.Module):
    def __init__(self, code_num, code_size,
                 adj, graph_size, hidden_size, t_attention_size, t_output_size,
                 output_size, dropout_rate, activation):
        super().__init__()
        self.classifier = Classifier(2613, output_size, dropout_rate, activation)
        # 双向LSTM捕获前后时序信息
        self.lstm = nn.LSTM(2613, hidden_size // 2, 2,
                            batch_first=True, bidirectional=True)
        # 时间注意力机制
        self.temporal_attention = TemporalAttention(hidden_size)
        self.selfattention = SelfAttention(size=200)

        # 使用高效图融合模块
        self.graph_fusion = EfficientGraphFusion(
            num_codes=code_num,
            code_size=code_size,
            hidden_size=100,
            num_heads=4
        )

        # 调整gated_fusion输入维度
        self.gated_fusion = EnhancedGatedTemporalFusion(
            input_dim=100,
            global_dim=200,
            hidden_dim=200,
            num_heads=4
        )

    def forward(self, adj, code_x, lens):
        code_x = code_x.float()

        packed = nn.utils.rnn.pack_padded_sequence(
            code_x, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=code_x.size(1)
        )  # (batch, seq_len, hidden_size)

        # Self-Attention增强时序依赖
        attn_out = self.selfattention(lstm_out)  # (batch, seq_len, hidden_size)

        # 时间注意力加权
        codes = self.temporal_attention(attn_out, lens)  # (batch, hidden_size)

        output = []
        for i, (code_x_i, len_i) in enumerate(zip(code_x, lens)):
            patient_seq = code_x_i[:len_i]  # (seq_len, code_size)
            # 动态图时空融合
            patient_rep = self.graph_fusion(patient_seq, adj)
            positions = torch.arange(len_i, device=device)  # 时间位置
            output_i = self.gated_fusion(patient_rep, codes[i], positions)

            output.append(output_i)
        output = torch.vstack(output)
        output = self.classifier(output)
        return output