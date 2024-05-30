import torch


@torch.no_grad()
def matcher_metrics(pred, data, prefix="", prefix_gt=None):
    def recall(m, gt_m):
        mask = (gt_m > -1).float()
        return ((m == gt_m) * mask).sum(1) / (1e-8 + mask.sum(1))

    def accuracy(m, gt_m):
        mask = (gt_m >= -1).float()
        return ((m == gt_m) * mask).sum(1) / (1e-8 + mask.sum(1))

    def precision(m, gt_m):
        mask = ((m > -1) & (gt_m >= -1)).float()
        return ((m == gt_m) * mask).sum(1) / (1e-8 + mask.sum(1))

    def ranking_ap(m, gt_m, scores):
        p_mask = ((m > -1) & (gt_m >= -1)).float()
        r_mask = (gt_m > -1).float()

        print(f"p_mask.shape: {p_mask.shape}")
        print(f"r_mask.shape: {r_mask.shape}")
        print(f"scores.shape: {scores.shape}")
        print(scores)
        # Adjust scores shape
        # Adjust scores shape if necessary
        if scores.dim() == 2:
            scores = scores.unsqueeze(1).expand(-1, p_mask.size(1), -1)


        # sort_ind = torch.argsort(-scores, dim=-1)
        # sort_ind = torch.argsort(-scores, dim=-1).unsqueeze(1).expand(-1, p_mask.size(1), -1)
        sort_ind = torch.argsort(-scores, dim=-1).unsqueeze(1).expand(-1, p_mask.size(1), -1)
        # sort_ind = torch.argsort(-scores, dim=-1).unsqueeze(1).expand(-1, p_mask.size(1), -1, -1)

        print(f"sort_ind.shape: {sort_ind.shape}")

        sorted_p_mask = torch.gather(p_mask, -1, sort_ind)
        sorted_r_mask = torch.gather(r_mask, -1, sort_ind)
        sorted_tp = torch.gather(m == gt_m, -1, sort_ind)

        print(f"sorted_p_mask.shape: {sorted_p_mask.shape}")
        print(f"sorted_r_mask.shape: {sorted_r_mask.shape}")
        print(f"sorted_tp.shape: {sorted_tp.shape}")

        p_pts = torch.cumsum(sorted_tp * sorted_p_mask, dim=-1) / (
                1e-8 + torch.cumsum(sorted_p_mask, dim=-1)
        )
        r_pts = torch.cumsum(sorted_tp * sorted_r_mask, dim=-1) / (
                1e-8 + sorted_r_mask.sum(dim=-1, keepdim=True)
        )
        r_pts_diff = r_pts[..., 1:] - r_pts[..., :-1]

        print(f"p_pts.shape: {p_pts.shape}")
        print(f"r_pts.shape: {r_pts.shape}")
        print(f"r_pts_diff.shape: {r_pts_diff.shape}")

        p_pts = p_pts[..., 1:]  # Align dimensions
        return torch.sum(r_pts_diff * p_pts, dim=-1)

        # return torch.sum(r_pts_diff * p_pts[..., 1:], dim=-1)
        # return torch.sum(r_pts_diff * p_pts[:, None, -1], dim=-1)

    if prefix_gt is None:
        prefix_gt = prefix
    print(f"here are the prediction keys for {prefix_gt}", pred.keys())
    # for key, value in pred:
    #     print(key, value)

    # Check shapes before calling ranking_ap
    scores = pred[f"{prefix}matching_scores0"]  # or however scores are derived
    print(f"scores.shape before call ap studd: {scores.shape}")

    # # Ensure the correct shape
    if scores.dim() == 2:  # If scores lack the num_pairs dimension
        scores = scores.unsqueeze(1).expand(-1, pred[f"{prefix}matches0"].size(1), -1)
        print("in matcher metrics had to change dim of scores again")


    print(f"scores.shape after reshape: {scores.shape}")

    rec = recall(pred[f"{prefix}matches0"], data[f"gt_{prefix_gt}matches0"])
    prec = precision(pred[f"{prefix}matches0"], data[f"gt_{prefix_gt}matches0"])
    acc = accuracy(pred[f"{prefix}matches0"], data[f"gt_{prefix_gt}matches0"])
    ap = ranking_ap(
        pred[f"{prefix}matches0"],
        data[f"gt_{prefix_gt}matches0"],
        scores,
        # pred[f"{prefix}matching_scores0"],
    )
    metrics = {
        f"{prefix}match_recall": rec,
        f"{prefix}match_precision": prec,
        f"{prefix}accuracy": acc,
        f"{prefix}average_precision": ap,
    }
    return metrics
