"""A message-passing graph neural network (GINE-style) implemented in plain PyTorch.

Atoms are nodes and bonds are (bidirectional) edges. Each layer updates an atom from the sum of
messages ReLU(h_u + W_e e_uv) sent by its neighbours, followed by an MLP, as in GIN with edge
features (Hu et al., 2020). A permutation-invariant readout pools atoms into a molecule vector
that an MLP head maps to the logit (classification) or the standardised pIC50 (regression).
"""
import copy

import numpy as np
import torch
from rdkit import Chem
from torch import nn

from .metrics import evaluate

ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]
HYBRID = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3]
CHIRAL = [Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW]
BONDS = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]


def one_hot(value, choices):
    """One-hot encoding with a trailing 'other' slot."""
    v = [0.0] * (len(choices) + 1)
    v[choices.index(value) if value in choices else -1] = 1.0
    return v


def atom_features(atom, use_chirality=True):
    f = (one_hot(atom.GetSymbol(), ELEMENTS) + one_hot(atom.GetDegree(), [0, 1, 2, 3, 4])
         + one_hot(atom.GetFormalCharge(), [-1, 0, 1]) + one_hot(atom.GetHybridization(), HYBRID)
         + one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3]) + [float(atom.GetIsAromatic()), float(atom.IsInRing())])
    if use_chirality:
        f += one_hot(atom.GetChiralTag(), CHIRAL)
    return f


def bond_features(bond):
    return one_hot(bond.GetBondType(), BONDS) + [float(bond.GetIsConjugated()), float(bond.IsInRing())]


EDGE_DIM = len(BONDS) + 1 + 2


def mol_to_graph(smiles, use_chirality=True):
    mol = Chem.MolFromSmiles(smiles)
    x = torch.tensor([atom_features(a, use_chirality) for a in mol.GetAtoms()], dtype=torch.float32)
    src, dst, ea = [], [], []
    for b in mol.GetBonds():
        i, j, f = b.GetBeginAtomIdx(), b.GetEndAtomIdx(), bond_features(b)
        src += [i, j]
        dst += [j, i]
        ea += [f, f]
    edge_index = torch.tensor([src, dst], dtype=torch.long).reshape(2, -1)
    edge_attr = torch.tensor(ea, dtype=torch.float32).reshape(-1, EDGE_DIM)
    return x, edge_index, edge_attr


def _ranges(starts, lengths):
    """Concatenation of arange(start, start + length) for each pair, without a Python loop."""
    offsets = np.cumsum(lengths) - lengths
    return torch.from_numpy(np.arange(lengths.sum()) - np.repeat(offsets, lengths) + np.repeat(starts, lengths))


class GraphSet:
    """All molecular graphs stored as flat tensors so that mini-batches are built by indexing."""

    def __init__(self, smiles, use_chirality=True):
        graphs = [mol_to_graph(s, use_chirality) for s in smiles]
        self.x = torch.cat([g[0] for g in graphs])
        self.edge_index = torch.cat([g[1] for g in graphs], 1)  # per-graph (local) atom indices
        self.edge_attr = torch.cat([g[2] for g in graphs])
        self.n_nodes = np.array([g[0].shape[0] for g in graphs])
        self.n_edges = np.array([g[1].shape[1] for g in graphs])
        self.node_ptr = np.concatenate([[0], np.cumsum(self.n_nodes)[:-1]])
        self.edge_ptr = np.concatenate([[0], np.cumsum(self.n_edges)[:-1]])
        self.atom_dim = self.x.shape[1]

    def __len__(self):
        return len(self.n_nodes)

    def batch(self, idx):
        idx = np.asarray(idx)
        nn_, ne = self.n_nodes[idx], self.n_edges[idx]
        nodes = _ranges(self.node_ptr[idx], nn_)
        edges = _ranges(self.edge_ptr[idx], ne)
        shift = torch.from_numpy(np.repeat(np.cumsum(nn_) - nn_, ne))
        batch = torch.from_numpy(np.repeat(np.arange(len(idx)), nn_))
        return self.x[nodes], self.edge_index[:, edges] + shift, self.edge_attr[edges], batch, len(idx)


class GINELayer(nn.Module):
    def __init__(self, dim, edge_dim, use_edges=True):
        super().__init__()
        self.edge = nn.Linear(edge_dim, dim) if use_edges else None
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, h, edge_index, edge_attr):
        src, dst = edge_index
        msg = h[src]
        if self.edge is not None:
            msg = msg + self.edge(edge_attr)
        agg = torch.zeros_like(h).index_add_(0, dst, torch.relu(msg))
        return self.mlp((1 + self.eps) * h + agg)


class MolGNN(nn.Module):
    def __init__(self, atom_dim, edge_dim, hidden=128, layers=3, dropout=0.1, readout="sum+mean+max",
                 use_edges=True, extra_dim=0):
        super().__init__()
        self.embed = nn.Linear(atom_dim, hidden)
        self.layers = nn.ModuleList([GINELayer(hidden, edge_dim, use_edges) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])
        self.readout = readout.split("+")
        self.dropout = nn.Dropout(dropout)
        in_head = hidden * len(self.readout) + extra_dim
        self.head = nn.Sequential(nn.Linear(in_head, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def embed_graph(self, x, edge_index, edge_attr, batch, n_graphs):
        h = self.embed(x)
        for layer, norm in zip(self.layers, self.norms):
            h = h + self.dropout(torch.relu(norm(layer(h, edge_index, edge_attr))))  # residual update
        pooled = []
        sums = torch.zeros(n_graphs, h.shape[1]).index_add_(0, batch, h)
        counts = torch.bincount(batch, minlength=n_graphs).clamp(min=1).unsqueeze(1).float()
        for r in self.readout:
            if r == "sum":
                pooled.append(sums)
            elif r == "mean":
                pooled.append(sums / counts)
            elif r == "max":
                out = torch.full((n_graphs, h.shape[1]), -1e9)
                pooled.append(out.scatter_reduce(0, batch.unsqueeze(1).expand_as(h), h, reduce="amax",
                                                 include_self=True))
        return torch.cat(pooled, 1)

    def forward(self, x, edge_index, edge_attr, batch, n_graphs, extra=None):
        z = self.embed_graph(x, edge_index, edge_attr, batch, n_graphs)
        if extra is not None:
            z = torch.cat([z, extra], 1)
        return self.head(z).squeeze(-1)


def _predict(model, graphs, idx, extra, batch_size=256):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), batch_size):
            b = idx[s:s + batch_size]
            x, ei, ea, bt, n = graphs.batch(b)
            out.append(model(x, ei, ea, bt, n, None if extra is None else extra[b]))
    return torch.cat(out).numpy()


def train_gnn(graphs, y, task, train_idx, val_idx, test_idx, seed=0, extra=None, hidden=128, layers=3,
              dropout=0.1, readout="sum+mean+max", use_edges=True, lr=1e-3, weight_decay=1e-5,
              batch_size=64, max_epochs=200, patience=30):
    """Train with early stopping on the validation primary metric; return test predictions."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    y = np.asarray(y, dtype=np.float32)
    if task == "reg":
        mu, sd = y[train_idx].mean(), y[train_idx].std()
    else:
        mu, sd = 0.0, 1.0
    y_t = torch.tensor((y - mu) / sd)
    extra_t = None
    if extra is not None:  # standardised molecule-level descriptors (statistics from the training set)
        e_mu, e_sd = np.nanmean(extra[train_idx], 0), np.nanstd(extra[train_idx], 0) + 1e-6
        e = np.nan_to_num((extra - e_mu) / e_sd)
        extra_t = torch.tensor(np.clip(e, -10, 10), dtype=torch.float32)

    model = MolGNN(graphs.atom_dim, EDGE_DIM, hidden, layers, dropout, readout, use_edges,
                   0 if extra is None else extra.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss() if task == "cls" else nn.MSELoss()

    def to_output(raw):
        return 1 / (1 + np.exp(-raw)) if task == "cls" else raw * sd + mu

    best, best_state, wait, history = None, None, 0, []
    for epoch in range(max_epochs):
        model.train()
        order = rng.permutation(train_idx)
        for s in range(0, len(order), batch_size):
            b = order[s:s + batch_size]
            if len(b) < 2:
                continue
            x, ei, ea, bt, n = graphs.batch(b)
            pred = model(x, ei, ea, bt, n, None if extra_t is None else extra_t[b])
            loss = loss_fn(pred, y_t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        val_pred = to_output(_predict(model, graphs, val_idx, extra_t))
        score = evaluate(task, y[val_idx], val_pred)["ROC_AUC" if task == "cls" else "RMSE"]
        score = score if task == "cls" else -score
        history.append(score)
        if best is None or score > best:
            best, best_state, wait = score, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    preds = {name: to_output(_predict(model, graphs, idx, extra_t))
             for name, idx in (("val", val_idx), ("test", test_idx))}
    return preds, {"epochs": epoch + 1, "best_val": best, "history": history}, model
