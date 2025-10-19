import os

import numpy as np


def save_sparse(path, x):
    idx = np.where(x > 0)
    values = x[idx]
    np.savez(path, idx=idx, values=values, shape=x.shape)


def load_sparse(path):
    data = np.load(path)
    idx, values = data['idx'], data['values']
    mat = np.zeros(data['shape'], dtype=values.dtype)
    mat[tuple(idx)] = values
    return mat


def save_data(path, code_x, visit_lens, codes_y, neighbors):
    save_sparse(os.path.join(path, 'code_x'), code_x)
    np.savez(os.path.join(path, 'visit_lens'), lens=visit_lens)
    save_sparse(os.path.join(path, 'code_y'), codes_y)
    # np.savez(os.path.join(path, 'hf_y'), hf_y=hf_y)
    # save_sparse(os.path.join(path, 'divided'), divided)
    save_sparse(os.path.join(path, 'neighbors'), neighbors)


def save_subgraphs(path, cooccur_subgraphs, temporal_subgraphs):
    """
    保存预计算的子图数据到指定路径
    
    参数:
        path: 保存路径
        cooccur_subgraphs: 共现子图列表 [(adj_matrix, node_indices), ...]
        temporal_subgraphs: 时序子图列表 [(adj_matrix, node_indices), ...]
    """
    cooccur_path = os.path.join(path, 'cooccur_subgraphs')
    temporal_path = os.path.join(path, 'temporal_subgraphs')
    
    # 确保目录存在
    os.makedirs(cooccur_path, exist_ok=True)
    os.makedirs(temporal_path, exist_ok=True)
    
    # 保存子图数量
    n_subgraphs = len(cooccur_subgraphs)
    np.savez(os.path.join(path, 'subgraph_count'), count=n_subgraphs)
    
    # 保存每个子图
    for i, ((cooccur_adj, cooccur_nodes), (temporal_adj, temporal_nodes)) in enumerate(zip(cooccur_subgraphs, temporal_subgraphs)):
        # 保存共现子图
        np.savez(
            os.path.join(cooccur_path, f'subgraph_{i}'),
            adj=cooccur_adj,
            nodes=cooccur_nodes
        )
        
        # 保存时序子图
        np.savez(
            os.path.join(temporal_path, f'subgraph_{i}'),
            adj=temporal_adj,
            nodes=temporal_nodes
        )
    
    print(f"Saved {n_subgraphs} subgraphs to {path}")


def load_subgraphs(path):
    """
    从指定路径加载预计算的子图数据
    
    参数:
        path: 子图数据路径
        
    返回:
        cooccur_subgraphs: 共现子图列表 [(adj_matrix, node_indices), ...]
        temporal_subgraphs: 时序子图列表 [(adj_matrix, node_indices), ...]
    """
    cooccur_path = os.path.join(path, 'cooccur_subgraphs')
    temporal_path = os.path.join(path, 'temporal_subgraphs')
    
    # 加载子图数量
    n_subgraphs = np.load(os.path.join(path, 'subgraph_count.npz'))['count']
    
    cooccur_subgraphs = []
    temporal_subgraphs = []
    
    # 加载每个子图
    for i in range(n_subgraphs):
        # 加载共现子图
        cooccur_data = np.load(os.path.join(cooccur_path, f'subgraph_{i}.npz'))
        cooccur_subgraphs.append((cooccur_data['adj'], cooccur_data['nodes']))
        
        # 加载时序子图
        temporal_data = np.load(os.path.join(temporal_path, f'subgraph_{i}.npz'))
        temporal_subgraphs.append((temporal_data['adj'], temporal_data['nodes']))
    
    print(f"Loaded {n_subgraphs} subgraphs from {path}")
    return cooccur_subgraphs, temporal_subgraphs
