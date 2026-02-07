def compute_iou(b1, b2):
    x1,y1,w1,h1 = b1
    x2,y2,w2,h2 = b2
    xa, ya = max(x1,x2), max(y1,y2)
    xb, yb = min(x1+w1,x2+w2), min(y1+h1,y2+h2)
    inter = max(0, xb-xa) * max(0, yb-ya)
    union = w1*h1 + w2*h2 - inter
    return inter/union if union else 0




def group_boxes(boxes, thresh=0.1):
    groups = []
    for b in boxes:
        placed = False
        for g in groups:
            if any(compute_iou(b, x) > thresh for x in g):
                g.append(b); placed = True; break
        if not placed:
            groups.append([b])
    return groups




def merge_boxes(group):
    xs = [b[0] for b in group]
    ys = [b[1] for b in group]
    xe = [b[0]+b[2] for b in group]
    ye = [b[1]+b[3] for b in group]
    return [min(xs), min(ys), max(xe)-min(xs), max(ye)-min(ys)]