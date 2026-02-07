import torch
import matplotlib.pyplot as plt

from utils.iou import group_boxes, merge_boxes

def crop_and_show(image, x, y, w, h, title="Prediction"):
    '''
    image: H x W x 3 (RGB)
    x, y, w, h: pixel coordinates
    '''

    x, y, w, h = map(int, [x, y, w, h])

    image = image.detach().cpu()
    image = image.permute(1, 2, 0)
    image = image.numpy()
    H, W, _ = image.shape

    crop = image[y:y+h, x:x+w]

    plt.figure(figsize=(3, 3))
    plt.imshow(crop)
    plt.title(title)
    plt.axis("off")
    plt.show()


@torch.no_grad()
def run_inference(
    model,
    test_loader,
    device,
    threshold=0.8,
    show_image=True,
):
    model.eval()
    processed = 0

    for images, targets in test_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # images: (1, P, 3, 48, 48)
        B, P, C, H, W = images.shape

        # --------------------------------------------------
        # Reconstruct original image from patches
        # --------------------------------------------------
        patches = images.reshape(B, 16, 16, C, H, W)
        patches = patches.permute(0, 3, 1, 4, 2, 5)
        orig_img = patches.reshape(B, C, 16 * H, 16 * W)

        img_np = orig_img[0].detach().cpu().permute(1, 2, 0).numpy()

        # --------------------------------------------------
        # Model forward
        # --------------------------------------------------
        flat_images = images.view(B * P, C, H, W)
        preds = model(flat_images)          # (P, 5)

        cls_logits = preds[:, 0]
        cls_probs = torch.sigmoid(cls_logits)
        reg_preds = preds[:, 1:5]

        # --------------------------------------------------
        # Collect positive boxes
        # --------------------------------------------------
        boxes = []
        for i in range(P):
            if cls_probs[i] > threshold:
                x, y, w, h = reg_preds[i].cpu().numpy()
                boxes.append([x, y, w, h])

        # --------------------------------------------------
        # Merge overlapping boxes iteratively
        # --------------------------------------------------
        merged_boxes = []
        if len(boxes) > 0:
            merged_boxes = boxes
            for _ in range(5):
                groups = group_boxes(merged_boxes, iou_thresh=0.01)
                merged_boxes = [merge_boxes(group) for group in groups]

        # --------------------------------------------------
        # Visualization
        # --------------------------------------------------
        if show_image and len(merged_boxes) > 0:
            plt.figure(figsize=(6, 6))
            plt.imshow(img_np)
            plt.title("Input Image")
            plt.axis("off")
            plt.show()

        for idx, (x, y, w, h) in enumerate(merged_boxes):
            crop_and_show(
                orig_img[0],
                x, y, w, h,
                title=f"Text region {idx + 1}"
            )

        processed += 1