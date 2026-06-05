import numpy as np
import torch

from rdkit import Chem
from torch_geometric.data import Data

from model import GINMultiTask


TARGETS = [
    "Egc","Egb","Eib","CED","Ei","Eea","nc","ne",
    "epse_6.0","epsc","epse_3.0","epse_1.78",
    "epse_15.0","epse_4.0","epse_5.0","epse_2.0",
    "epse_9.0","epse_7.0","TSb","TSy","epsb","YM",
    "permCH4","permCO2","permH2","permO2",
    "permN2","permHe","Eat","rho","LOI",
    "Xc","Xe","Cp","Td","Tg","Tm"
]


def atom_features(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        atom.GetTotalValence(),
        int(atom.GetIsAromatic())
    ]


def smiles_to_graph(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Invalid SMILES")

    x = torch.tensor(
        [atom_features(a) for a in mol.GetAtoms()],
        dtype=torch.float
    )

    edges = []

    for bond in mol.GetBonds():

        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edges.append([i, j])
        edges.append([j, i])

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges).t().contiguous()

    data = Data(
        x=x,
        edge_index=edge_index
    )

    data.batch = torch.zeros(
        x.size(0),
        dtype=torch.long
    )

    return data


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = GINMultiTask(
    in_channels=4,
    hidden_dim=128,
    num_layers=3,
    num_tasks=37,
    dropout=0.1
)

model.load_state_dict(
    torch.load(
        "best_gnn.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

stats = np.load("target_norm_stats.npz")

target_means = stats["mean"]
target_stds = stats["std"]


def predict(smiles):

    graph = smiles_to_graph(smiles)
    graph = graph.to(device)

    with torch.no_grad():

        pred = model(graph)

        pred = pred.cpu().numpy()[0]

        pred_denorm = (
            pred * target_stds
            + target_means
        )

    return {
        name: float(value)
        for name, value in zip(
            TARGETS,
            pred_denorm
        )
    }