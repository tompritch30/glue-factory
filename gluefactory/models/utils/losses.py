import torch
import torch.nn as nn
from omegaconf import OmegaConf


def weight_loss(log_assignment, weights, gamma=0.0):
    b, m, n = log_assignment.shape
    m -= 1
    n -= 1

    loss_sc = log_assignment * weights

    num_neg0 = weights[:, :m, -1].sum(-1).clamp(min=1.0)
    num_neg1 = weights[:, -1, :n].sum(-1).clamp(min=1.0)
    num_pos = weights[:, :m, :n].sum((-1, -2)).clamp(min=1.0)

    nll_pos = -loss_sc[:, :m, :n].sum((-1, -2))
    nll_pos /= num_pos.clamp(min=1.0)

    nll_neg0 = -loss_sc[:, :m, -1].sum(-1)
    nll_neg1 = -loss_sc[:, -1, :n].sum(-1)

    nll_neg = (nll_neg0 + nll_neg1) / (num_neg0 + num_neg1)

    return nll_pos, nll_neg, num_pos, (num_neg0 + num_neg1) / 2.0


class NLLLoss(nn.Module):
    default_conf = {
        "nll_balancing": 0.5,
        "gamma_f": 0.0,  # focal loss
    }

    def __init__(self, conf):
        super().__init__()
        self.conf = OmegaConf.merge(self.default_conf, conf)
        self.loss_fn = self.nll_loss

    def forward(self, pred, data, weights=None):
        log_assignment = pred["log_assignment"]
        if weights is None:
            weights = self.loss_fn(log_assignment, data)
        nll_pos, nll_neg, num_pos, num_neg = weight_loss(
            log_assignment, weights, gamma=self.conf.gamma_f
        )
        nll = (
            self.conf.nll_balancing * nll_pos + (1 - self.conf.nll_balancing) * nll_neg
        )

        return (
            nll,
            weights,
            {
                "assignment_nll": nll,
                "nll_pos": nll_pos,
                "nll_neg": nll_neg,
                "num_matchable": num_pos,
                "num_unmatchable": num_neg,
            },
        )

    def nll_loss(self, log_assignment, data):
        m, n = data["gt_matches0"].size(-1), data["gt_matches1"].size(-1)
        positive = data["gt_assignment"].float()
        neg0 = (data["gt_matches0"] == -1).float()
        neg1 = (data["gt_matches1"] == -1).float()

        print("m, n", m, n)
        print(f"log_assignment.shape: {log_assignment.shape}")
        print(f"positive.shape: {positive.shape}")
        print(f"negative0.shape: {neg0.shape}")
        print(f"negative1.shape: {neg1.shape}")

        weights = torch.zeros_like(log_assignment)

        # Manually adjust the shape of positive to match (batch_size, m, n)
        batch_size = positive.shape[0]
        new_positive = torch.zeros((batch_size, m, n), device=positive.device)

        # Copy values from positive to new_positive
        for i in range(batch_size):
            new_positive[i, :positive.shape[2], :positive.shape[3]] = positive[i, 0, :, :]

        positive = new_positive
        print(f"Adjusted positive.shape: {positive.shape}")

        weights[:, :m, :n] = positive

        # Manually adjust the shape of neg0 and neg1 to match (batch_size, m, 1) and (batch_size, 1, n)
        new_neg0 = torch.zeros((batch_size, m, 1), device=neg0.device)
        new_neg1 = torch.zeros((batch_size, 1, n), device=neg1.device)

        # Copy values from neg0 and neg1 to new_neg0 and new_neg1
        for i in range(batch_size):
            new_neg0[i, :, 0] = neg0[i, 0, :]
            new_neg1[i, 0, :] = neg1[i, 0, :]

        neg0 = new_neg0
        neg1 = new_neg1

        print(f"Adjusted neg0.shape: {neg0.shape}")
        print(f"Adjusted neg1.shape: {neg1.shape}")

        weights[:, :m, -1] = neg0.squeeze(-1)
        weights[:, -1, :n] = neg1.squeeze(1)
        print(f"weights.shape after assignment: {weights.shape}")

        return weights

    # def nll_loss(self, log_assignment, data):
    #     m, n = data["gt_matches0"].size(-1), data["gt_matches1"].size(-1)
    #     positive = (data["gt_assignment"].float())
    #     print(f"positive.shape BEFORE: {positive.shape}")
    #     # positive.squeeze(1)
    #     neg0 = (data["gt_matches0"] == -1).float()
    #     neg1 = (data["gt_matches1"] == -1).float()
    #
    #     # positive.squeeze(0)
    #     # Adjust the shape of positive to match the required shape
    #     if positive.dim() > 3:
    #         positive = positive.squeeze(1)  # This removes the second dimension if it's size 1
    #         print("positive squeeze!!")
    #         if positive.dim() > 3:  # If there's still an extra dimension, remove it
    #             print("positive squeeze!!")
    #             positive = positive.squeeze(1)
    #     # if positive.dim() > 3:
    #     #
    #     #     positive = positive.squeeze(3)
    #
    #     print("m, n", m, n)
    #     # m, n = positive.shape[1:3]
    #     print(f"log_assignment.shape: {log_assignment.shape}")
    #     print(f"positive.shape: {positive.shape}")
    #     print(f"negative0.shape: {neg0.shape}")
    #     print(f"negative1.shape: {neg1.shape}")
    #
    #     weights = torch.zeros_like(log_assignment)
    #     ### Ensure positive has compatible shape before assignment
    #     if positive.shape[1] != m or positive.shape[2] != n:
    #         print("Adjusting positive shape for compatibility")
    #         positive = positive[:, :m, :n]
    #
    #     print(f"positive.shape: {positive.shape} ")
    #     print("the weights: " ,weights[:, :m, :n], "\n\n\n\n\n")
    #     print("here is positive", positive, "\n\n\n\n\n ---------------------------")
    #     weights[:, :m, :n] = positive
    #
    #     # Ensure neg0 and neg1 have compatible shapes before assignment
    #     if neg0.shape[-1] < weights.shape[-1]:
    #         print("neg0.shape[-1] < weights.shape[-1]")
    #         neg0 = neg0.expand(weights.shape[:-1])
    #     if neg1.shape[-1] < weights.shape[-1]:
    #         print("neg1.shape[-1] < weights.shape[-1]")
    #         neg1 = neg1.expand(weights.shape[:-1])
    #
    #     weights[:, :m, -1] = neg0
    #     weights[:, -1, :n] = neg1
    #     print(f"weights.shape after assignment: {weights.shape}")
    #
    #     return weights
