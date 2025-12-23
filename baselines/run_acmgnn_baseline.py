"""
Phase 1: Establish the True Bar - Run ACM-GNN on Actor and Squirrel
Using Geom-GCN standard splits (10 folds) for fair comparison

ACM-GNN is designed for heterophilic graphs and can serve as a strong teacher.

Target Performance:
- Actor: > 37.5% (SOTA teacher)
- Squirrel: > 40%
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
import json
import argparse
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ACMGCN(nn.Module):
    """
    ACM-GCN: Adaptive Channel Mixing GCN
    
    Key idea: Learn to adaptively mix low-pass and high-pass filtered signals
    for handling both homophilic and heterophilic graphs.
    
    Based on the official implementation from:
    https://github.com/SitaoLuan/ACM-GNN
    """
    def __init__(self, nfeat, nhid, nclass, dropout, nlayers=2, variant=False):
        super(ACMGCN, self).__init__()
        
        self.dropout = dropout
        self.nlayers = nlayers
        self.variant = variant
        
        # Three-channel weights for first layer
        self.weight_low = nn.Parameter(torch.FloatTensor(nfeat, nhid))
        self.weight_high = nn.Parameter(torch.FloatTensor(nfeat, nhid))
        self.weight_mlp = nn.Parameter(torch.FloatTensor(nfeat, nhid))
        
        # Attention vectors for channel mixing
        self.att_vec_low = nn.Parameter(torch.FloatTensor(nhid, 1))
        self.att_vec_high = nn.Parameter(torch.FloatTensor(nhid, 1))
        self.att_vec_mlp = nn.Parameter(torch.FloatTensor(nhid, 1))
        self.att_vec = nn.Parameter(torch.FloatTensor(3, 3))
        
        # Second layer weights
        self.weight_low2 = nn.Parameter(torch.FloatTensor(nhid, nclass))
        self.weight_high2 = nn.Parameter(torch.FloatTensor(nhid, nclass))
        self.weight_mlp2 = nn.Parameter(torch.FloatTensor(nhid, nclass))
        
        self.att_vec_low2 = nn.Parameter(torch.FloatTensor(nclass, 1))
        self.att_vec_high2 = nn.Parameter(torch.FloatTensor(nclass, 1))
        self.att_vec_mlp2 = nn.Parameter(torch.FloatTensor(nclass, 1))
        self.att_vec2 = nn.Parameter(torch.FloatTensor(3, 3))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight_mlp.size(1))
        std_att = 1. / np.sqrt(self.att_vec_mlp.size(1))
        std_att_vec = 1. / np.sqrt(self.att_vec.size(1))
        
        self.weight_low.data.uniform_(-stdv, stdv)
        self.weight_high.data.uniform_(-stdv, stdv)
        self.weight_mlp.data.uniform_(-stdv, stdv)
        self.weight_low2.data.uniform_(-stdv, stdv)
        self.weight_high2.data.uniform_(-stdv, stdv)
        self.weight_mlp2.data.uniform_(-stdv, stdv)
        
        self.att_vec_low.data.uniform_(-std_att, std_att)
        self.att_vec_high.data.uniform_(-std_att, std_att)
        self.att_vec_mlp.data.uniform_(-std_att, std_att)
        self.att_vec_low2.data.uniform_(-std_att, std_att)
        self.att_vec_high2.data.uniform_(-std_att, std_att)
        self.att_vec_mlp2.data.uniform_(-std_att, std_att)
        
        self.att_vec.data.uniform_(-std_att_vec, std_att_vec)
        self.att_vec2.data.uniform_(-std_att_vec, std_att_vec)
        
    def attention(self, output_low, output_high, output_mlp, att_vec_low, att_vec_high, att_vec_mlp, att_vec):
        """Compute adaptive attention weights for three channels"""
        T = 3  # Temperature
        logits = torch.mm(
            torch.sigmoid(torch.cat([
                torch.mm(output_low, att_vec_low),
                torch.mm(output_high, att_vec_high),
                torch.mm(output_mlp, att_vec_mlp)
            ], dim=1)),
            att_vec
        ) / T
        att = torch.softmax(logits, dim=1)
        return att[:, 0:1], att[:, 1:2], att[:, 2:3]
        
    def forward(self, x, adj_low, adj_high):
        """
        Args:
            x: Node features [N, F]
            adj_low: Normalized adjacency (low-pass filter) - sparse
            adj_high: I - adj_low (high-pass filter) - sparse
        """
        x = F.dropout(x, self.dropout, training=self.training)
        
        # First layer: three-channel convolution
        output_low = F.relu(torch.spmm(adj_low, torch.mm(x, self.weight_low)))
        output_high = F.relu(torch.spmm(adj_high, torch.mm(x, self.weight_high)))
        output_mlp = F.relu(torch.mm(x, self.weight_mlp))
        
        # Adaptive channel mixing
        att_low, att_high, att_mlp = self.attention(
            output_low, output_high, output_mlp,
            self.att_vec_low, self.att_vec_high, self.att_vec_mlp, self.att_vec
        )
        h = 3 * (att_low * output_low + att_high * output_high + att_mlp * output_mlp)
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        # Second layer
        output_low2 = F.relu(torch.spmm(adj_low, torch.mm(h, self.weight_low2)))
        output_high2 = F.relu(torch.spmm(adj_high, torch.mm(h, self.weight_high2)))
        output_mlp2 = F.relu(torch.mm(h, self.weight_mlp2))
        
        att_low2, att_high2, att_mlp2 = self.attention(
            output_low2, output_high2, output_mlp2,
            self.att_vec_low2, self.att_vec_high2, self.att_vec_mlp2, self.att_vec2
        )
        out = 3 * (att_low2 * output_low2 + att_high2 * output_high2 + att_mlp2 * output_mlp2)
        
        return F.log_softmax(out, dim=1)


def load_geom_gcn_data(dataset_name, split_idx=0):
    """Load heterophilic dataset with Geom-GCN standard splits."""
    from torch_geometric.datasets import Actor, WikipediaNetwork
    
    if dataset_name == 'actor':
        dataset = Actor(root='./data/actor')
    elif dataset_name == 'squirrel':
        dataset = WikipediaNetwork(root='./data/heterophilic', name='Squirrel', 
                                   geom_gcn_preprocess=True)
    elif dataset_name == 'chameleon':
        dataset = WikipediaNetwork(root='./data/heterophilic', name='Chameleon',
                                   geom_gcn_preprocess=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    data = dataset[0]
    
    features = data.x.numpy()
    labels = data.y.numpy()
    num_nodes = features.shape[0]
    
    edge_index = data.edge_index.numpy()
    adj = sp.coo_matrix(
        (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
        shape=(num_nodes, num_nodes)
    )
    
    if hasattr(data, 'train_mask') and data.train_mask.dim() == 2:
        split_idx = split_idx % data.train_mask.shape[1]
        train_mask = data.train_mask[:, split_idx].numpy()
        val_mask = data.val_mask[:, split_idx].numpy()
        test_mask = data.test_mask[:, split_idx].numpy()
    else:
        train_mask = data.train_mask.numpy()
        val_mask = data.val_mask.numpy()
        test_mask = data.test_mask.numpy()
    
    idx_train = np.where(train_mask)[0]
    idx_val = np.where(val_mask)[0]
    idx_test = np.where(test_mask)[0]
    
    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(adj):
    """Symmetric normalization: D^{-1/2} A D^{-1/2}"""
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalize_features(features):
    """Row-normalize features"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert scipy sparse matrix to torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    return correct.sum() / len(labels)


def run_acmgnn_experiment(dataset_name, config, num_splits=10, use_cuda=False, 
                          save_teacher=False, save_dir=None):
    """
    Run ACM-GNN on a dataset with multiple splits.
    
    Args:
        dataset_name: 'actor' or 'squirrel'
        config: hyperparameter configuration
        num_splits: number of splits to run
        use_cuda: whether to use GPU
        save_teacher: whether to save teacher model and logits
        save_dir: directory to save teacher outputs
    
    Returns:
        mean_acc, std_acc, all_results
    """
    results = []
    
    for split_idx in range(num_splits):
        print(f"\n--- Split {split_idx} ---")
        
        # Load data
        adj, features, labels, idx_train, idx_val, idx_test = load_geom_gcn_data(
            dataset_name, split_idx
        )
        
        num_nodes = features.shape[0]
        
        # Normalize adjacency: D^{-1/2} (A + I) D^{-1/2}
        adj_low = normalize_adj(adj)
        adj_low = sparse_mx_to_torch_sparse_tensor(adj_low)
        
        # High-pass filter: I - adj_low
        adj_high_dense = sp.eye(num_nodes) - adj_low.to_dense().numpy()
        adj_high = torch.FloatTensor(adj_high_dense).to_sparse()
        
        features = normalize_features(sp.csr_matrix(features))
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
        
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        
        if use_cuda:
            features = features.cuda()
            adj_low = adj_low.cuda()
            adj_high = adj_high.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()
            idx_val = idx_val.cuda()
            idx_test = idx_test.cuda()
        
        # Initialize model
        model = ACMGCN(
            nfeat=features.shape[1],
            nhid=config['hidden'],
            nclass=int(labels.max().item()) + 1,
            dropout=config['dropout'],
            nlayers=config['nlayers']
        )
        
        if use_cuda:
            model.cuda()
        
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], 
                              weight_decay=config['weight_decay'])
        
        # Training
        best_val_acc = 0
        best_test_acc = 0
        best_model_state = None
        best_logits = None
        patience_counter = 0
        
        for epoch in range(config['epochs']):
            model.train()
            optimizer.zero_grad()
            output = model(features, adj_low, adj_high)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                output = model(features, adj_low, adj_high)
            val_acc = accuracy(output[idx_val], labels[idx_val]).item()
            test_acc = accuracy(output[idx_test], labels[idx_test]).item()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_logits = output.detach().cpu()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= config['patience']:
                break
        
        print(f"Split {split_idx}: Test Acc = {best_test_acc*100:.2f}%")
        results.append(best_test_acc)
        
        # Save teacher model and logits for distillation
        if save_teacher and save_dir:
            split_dir = os.path.join(save_dir, f'split_{split_idx}')
            os.makedirs(split_dir, exist_ok=True)
            
            torch.save(best_model_state, os.path.join(split_dir, 'teacher_model.pth'))
            torch.save(best_logits, os.path.join(split_dir, 'teacher_logits.pt'))
            
            # Also save soft labels (probabilities)
            soft_labels = F.softmax(best_logits, dim=1)
            torch.save(soft_labels, os.path.join(split_dir, 'teacher_soft_labels.pt'))
            
            print(f"  Saved teacher model and logits to {split_dir}")
    
    mean_acc = np.mean(results) * 100
    std_acc = np.std(results) * 100
    
    return mean_acc, std_acc, results


# ACM-GNN hyperparameters (tuned for heterophilic graphs)
# Based on ACM-GNN paper settings
ACMGNN_CONFIGS = {
    'actor': {
        'hidden': 64,
        'lr': 0.01,
        'dropout': 0.0,  # Lower dropout for heterophilic
        'weight_decay': 0.0,
        'nlayers': 2,
        'epochs': 1000,
        'patience': 200
    },
    'squirrel': {
        'hidden': 64,
        'lr': 0.01,
        'dropout': 0.0,
        'weight_decay': 0.0,
        'nlayers': 2,
        'epochs': 1000,
        'patience': 200
    },
    'chameleon': {
        'hidden': 64,
        'lr': 0.01,
        'dropout': 0.0,
        'weight_decay': 0.0,
        'nlayers': 2,
        'epochs': 1000,
        'patience': 200
    }
}


def main():
    parser = argparse.ArgumentParser(description='Run ACM-GNN Baseline')
    parser.add_argument('--dataset', type=str, default='actor', 
                       choices=['actor', 'squirrel', 'chameleon'])
    parser.add_argument('--num_splits', type=int, default=10)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_teacher', action='store_true',
                       help='Save teacher model and logits for distillation')
    args = parser.parse_args()
    
    print("="*60)
    print(f"Running ACM-GNN Baseline on {args.dataset.upper()}")
    print(f"Using Geom-GCN standard splits ({args.num_splits} folds)")
    print("="*60)
    
    config = ACMGNN_CONFIGS[args.dataset]
    print(f"\nConfig: {config}")
    
    # Setup save directory for teacher
    save_dir = None
    if args.save_teacher:
        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'checkpoints', 
            f'acmgnn_teacher_{args.dataset}'
        )
        os.makedirs(save_dir, exist_ok=True)
    
    mean_acc, std_acc, all_results = run_acmgnn_experiment(
        args.dataset, config, args.num_splits, args.cuda,
        save_teacher=args.save_teacher, save_dir=save_dir
    )
    
    print("\n" + "="*60)
    print(f"RESULTS: {args.dataset.upper()}")
    print(f"Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"All splits: {[f'{r*100:.2f}' for r in all_results]}")
    print("="*60)
    
    # Save results
    if args.save_results:
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        results_file = os.path.join(results_dir, f'acmgnn_baseline_{args.dataset}.json')
        with open(results_file, 'w') as f:
            json.dump({
                'dataset': args.dataset,
                'model': 'ACM-GNN+',
                'mean_acc': mean_acc,
                'std_acc': std_acc,
                'all_results': all_results,
                'config': config,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
