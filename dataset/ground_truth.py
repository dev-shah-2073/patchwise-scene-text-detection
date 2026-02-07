import torch


PATCH_SIZE = 48
TRANSFORM_SIZE = 768




def modify_coords(coords):
    out = []
    for c in coords:
        x = min(c[0], c[3])
        y = min(c[4], c[5])
        w = max(c[1], c[2]) - x
        h = max(c[6], c[7]) - y
        out.append((x, y, w, h))
    return torch.tensor(out)




def generate_ground_truth(coords):
    gt = []
    grid = TRANSFORM_SIZE // PATCH_SIZE


    for j in range(grid):
        for i in range(grid):
            x1, y1 = i * PATCH_SIZE, j * PATCH_SIZE
            flag = 1
            
            for c in coords:
                if x1 >= c[0] and y1 >= c[1] and x1 <= c[0]+c[2] and y1 <= c[1]+c[3]:
                    inter = (min(x1+PATCH_SIZE, c[0]+c[2]) - x1) * \
                    (min(y1+PATCH_SIZE, c[1]+c[3]) - y1)
                    if inter / (PATCH_SIZE * PATCH_SIZE) >= 0.25:
                        gt.append((1, c[0], c[1], c[2], c[3]))
                        flag = 0
                        break

            if flag:
                gt.append((0, 0, 0, 0, 0))

    return torch.tensor(gt)