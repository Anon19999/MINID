import torch

NUMBER_OF_BATCH_PER_LOG = 100
SEED = 12
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
