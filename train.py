import torch
from training.trainer import train
from dataset.dataloader import build_dataloaders
from model.mobilenet import build_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = "Data"

train_loader, test_loader = build_dataloaders(root_dir=data_dir, batch_size=16, num_workers=2, pin_memory=True)

model = build_model()
train(model, train_loader, device)