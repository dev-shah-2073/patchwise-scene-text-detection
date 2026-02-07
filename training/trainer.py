import torch
from model.loss import ConditionalMultiTaskLoss


def train(model, loader, device, epochs=1000, lr=0.002):
    model.to(device)
    model.train()


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = ConditionalMultiTaskLoss()


    for epoch in range(epochs):
        total = 0
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            B, P, C, H, W = imgs.shape


            imgs = imgs.view(B*P, C, H, W)
            targets = targets.view(B*P, 5)


            preds = model(imgs)
            loss = criterion(preds, targets)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            total += loss.item()


        print(f"Epoch {epoch+1}: Loss = {total:.2f}")