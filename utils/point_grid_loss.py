import torch
import torch.nn as nn
import torch.nn.functional as F


class PointGridLoss(nn.Module):
    """
    True point-based grid supervision.
    Counts how many GT points fall inside each grid cell.
    """

    def __init__(self, level=1):
        super().__init__()
        self.level = level  # GAME level

    def forward(self, pred_density, points_list, img_hw):
        """
        pred_density: [B, 1, H, W]
        points_list: list of tensors [Ni, 2] in (x, y) image coords
        img_hw: (H_img, W_img)
        """
        B, _, H, W = pred_density.shape
        H_img, W_img = img_hw

        cells = 2 ** self.level
        loss = 0.0

        for b in range(B):
            pred = pred_density[b, 0]
            points = points_list[b]

            if points.numel() == 0:
                continue

            # Scale points to density map resolution
            px = points[:, 0] * (W / W_img)
            py = points[:, 1] * (H / H_img)

            h_step = H // cells
            w_step = W // cells

            for i in range(cells):
                for j in range(cells):
                    h0, h1 = i * h_step, (i + 1) * h_step
                    w0, w1 = j * w_step, (j + 1) * w_step

                    # Count GT points in this cell
                    mask = (
                        (px >= w0) & (px < w1) &
                        (py >= h0) & (py < h1)
                    )
                    gt_count = mask.sum().float()

                    pred_count = pred[h0:h1, w0:w1].sum()
                    loss += F.l1_loss(pred_count, gt_count)

        return loss / B
