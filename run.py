import torch
from inference.predict import run_inference
from dataset.dataloader import build_dataloaders
from model.mobilenet import build_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = "Data"

train_loader, test_loader = build_dataloaders(root_dir=data_dir, batch_size=16, num_workers=2, pin_memory=True)

model = build_model()

run_inference(
    model=model,
    test_loader=test_loader,
    device=device,
    threshold=0.8,
    max_images=1
)