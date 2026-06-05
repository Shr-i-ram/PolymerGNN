import torch
from model import GINMultiTask

device = "cuda" if torch.cuda.is_available() else "cpu"

model = GINMultiTask(
    in_channels=4,
    hidden_dim=128,
    num_layers=3,
    num_tasks=37,
    dropout=0.1
)

state = torch.load(
    "best_gnn.pth",
    map_location=device
)

model.load_state_dict(state)

print("MODEL LOADED")