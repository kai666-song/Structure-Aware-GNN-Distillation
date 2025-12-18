import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import sys
import torch

# PyTorch Geometric imports for various datasets
try:
    from torch_geometric.datasets import Amazon, Coauthor, WikipediaNetwork, Actor
    from torch_geometric.transforms import RandomNodeSplit
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("Warning: torch_geometric not installed. Extended datasets unavailable.")

# OGB imports
try:
    from ogb.nodeproppred import PygNodePropPredDataset
    HAS_OGB = True
except ImportError:
    HAS_OGB = False
    print("Warning: ogb not installed. OGB datasets unavailable.")


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool_)


def load_data_new(dataset_str):
    """
    Loads input data from gcn/data directory or PyG datasets.
    
    Supports: 
    - Planetoid: cora, citeseer, pubmed
    - Amazon: amazon-computers, amazon-photo
    - Coauthor: coauthor-cs, coauthor-physics
    - Heterophilic: chameleon, squirrel, actor
    - OGB: ogbn-arxiv
    """
    dataset_str = dataset_str.lower()
    
    # Check dataset type
    if dataset_str in ['amazon-computers', 'amazon-photo', 'computers', 'photo']:
        return load_amazon_data(dataset_str)
    elif dataset_str in ['coauthor-cs', 'coauthor-physics', 'cs', 'physics']:
        return load_coauthor_data(dataset_str)
    elif dataset_str in ['chameleon', 'squirrel']:
        return load_heterophilic_data(dataset_str)
    elif dataset_str in ['actor']:
        return load_actor_data()
    elif dataset_str in ['ogbn-arxiv', 'arxiv']:
        return load_ogb_arxiv()
    else:
        # Original loading logic for cora, citeseer, pubmed
        return load_planetoid_data(dataset_str)


def load_amazon_data(dataset_str):
    """Load Amazon dataset (Computers or Photo) using PyTorch Geometric."""
    if not HAS_PYG:
        raise ImportError("torch_geometric required for Amazon datasets. "
                         "Install with: pip install torch_geometric")
    
    # Map dataset name
    if dataset_str in ['amazon-computers', 'computers']:
        name = 'Computers'
    else:
        name = 'Photo'
    
    # Load dataset with random split
    transform = RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0.2)
    dataset = Amazon(root='./data/amazon', name=name, transform=transform)
    data = dataset[0]
    
    return _convert_pyg_to_format(data)


def load_coauthor_data(dataset_str):
    """Load Coauthor dataset (CS or Physics) using PyTorch Geometric."""
    if not HAS_PYG:
        raise ImportError("torch_geometric required for Coauthor datasets. "
                         "Install with: pip install torch_geometric")
    
    # Map dataset name
    if dataset_str in ['coauthor-cs', 'cs']:
        name = 'CS'
    else:
        name = 'Physics'
    
    # Load dataset with random split
    transform = RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0.2)
    dataset = Coauthor(root='./data/coauthor', name=name, transform=transform)
    data = dataset[0]
    
    return _convert_pyg_to_format(data)


def _convert_pyg_to_format(data):
    """Convert PyG data object to the format expected by the training code."""
    # Get features and labels
    features = data.x.numpy()
    labels = data.y.numpy()
    num_nodes = features.shape[0]
    num_classes = int(labels.max()) + 1
    
    # Build adjacency matrix from edge_index
    edge_index = data.edge_index.numpy()
    adj = sp.coo_matrix(
        (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
        shape=(num_nodes, num_nodes)
    )
    adj = adj.tocsr()
    
    # Get masks
    train_mask = data.train_mask.numpy()
    val_mask = data.val_mask.numpy()
    test_mask = data.test_mask.numpy()
    
    idx_train = np.where(train_mask)[0]
    idx_val = np.where(val_mask)[0]
    idx_test = np.where(test_mask)[0]
    
    # Create one-hot labels
    labels_onehot = np.zeros((num_nodes, num_classes))
    labels_onehot[np.arange(num_nodes), labels] = 1
    
    y_train = np.zeros(labels_onehot.shape)
    y_val = np.zeros(labels_onehot.shape)
    y_test = np.zeros(labels_onehot.shape)
    y_train[train_mask, :] = labels_onehot[train_mask, :]
    y_val[val_mask, :] = labels_onehot[val_mask, :]
    y_test[test_mask, :] = labels_onehot[test_mask, :]
    
    # Convert to sparse features
    features = sp.lil_matrix(features)
    
    # Convert to torch tensors
    labels_tensor = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    return adj, features, labels_tensor, y_train, y_val, y_test, train_mask, val_mask, test_mask, idx_train, idx_val, idx_test


def load_heterophilic_data(dataset_str):
    """Load heterophilic datasets (Chameleon, Squirrel) using PyTorch Geometric.
    
    These are important for testing generalization on non-homophilic graphs.
    """
    if not HAS_PYG:
        raise ImportError("torch_geometric required for heterophilic datasets.")
    
    # WikipediaNetwork contains Chameleon and Squirrel
    name = dataset_str.capitalize()
    transform = RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0.2)
    dataset = WikipediaNetwork(root='./data/heterophilic', name=name, transform=transform)
    data = dataset[0]
    
    return _convert_pyg_to_format(data)


def load_actor_data():
    """Load Actor dataset (heterophilic) using PyTorch Geometric."""
    if not HAS_PYG:
        raise ImportError("torch_geometric required for Actor dataset.")
    
    transform = RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0.2)
    dataset = Actor(root='./data/actor', transform=transform)
    data = dataset[0]
    
    return _convert_pyg_to_format(data)


def load_ogb_arxiv():
    """Load OGB-Arxiv dataset (large-scale graph).
    
    This is critical for proving scalability and real-world applicability.
    ~170k nodes, ~1.2M edges, 40 classes
    """
    if not HAS_OGB:
        raise ImportError("ogb required for OGB datasets. Install with: pip install ogb")
    
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./data/ogb')
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    
    # Get features and labels
    features = data.x.numpy()
    labels = data.y.squeeze().numpy()
    num_nodes = features.shape[0]
    num_classes = int(labels.max()) + 1
    
    # Build adjacency matrix from edge_index
    edge_index = data.edge_index.numpy()
    adj = sp.coo_matrix(
        (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
        shape=(num_nodes, num_nodes)
    )
    adj = adj.tocsr()
    
    # Get split indices
    idx_train = split_idx['train'].numpy()
    idx_val = split_idx['valid'].numpy()
    idx_test = split_idx['test'].numpy()
    
    # Create masks
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True
    
    # Create one-hot labels
    labels_onehot = np.zeros((num_nodes, num_classes))
    labels_onehot[np.arange(num_nodes), labels] = 1
    
    y_train = np.zeros(labels_onehot.shape)
    y_val = np.zeros(labels_onehot.shape)
    y_test = np.zeros(labels_onehot.shape)
    y_train[train_mask, :] = labels_onehot[train_mask, :]
    y_val[val_mask, :] = labels_onehot[val_mask, :]
    y_test[test_mask, :] = labels_onehot[test_mask, :]
    
    # Convert to sparse features
    features = sp.lil_matrix(features)
    
    # Convert to torch tensors
    labels_tensor = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    print(f"OGB-Arxiv loaded: {num_nodes} nodes, {edge_index.shape[1]} edges, {num_classes} classes")
    print(f"Train/Val/Test: {len(idx_train)}/{len(idx_val)}/{len(idx_test)}")
    
    return adj, features, labels_tensor, y_train, y_val, y_test, train_mask, val_mask, test_mask, idx_train, idx_val, idx_test


def load_planetoid_data(dataset_str):
    """
    Loads input data from gcn/data directory (original Planetoid format).
    
    Supports: cora, citeseer, pubmed
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("./data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("./data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    
    if dataset_str == 'citeseer':
        one_hot = torch.LongTensor(labels)
        labels = torch.argmax(one_hot, dim=1)
    else:
        labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, y_train, y_val, y_test, train_mask, val_mask, test_mask, idx_train, idx_val, idx_test


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation."""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
