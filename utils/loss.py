import torch
import torch.nn.functional as F
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = alpha #weight parameter will act as the alpha parameter to balance class weights
        self.reduction = reduction

    def forward(self, input, target):
        if self.weight is not None:
            # alpha_factor = torch.ones(target.size()) * self.weight
            # alpha_factor = torch.where(torch.eq(target, 1), 1.0/alpha_factor, alpha_factor/alpha_factor)
            # ce_loss = F.cross_entropy(input, target, reduction='none', weight=None)
            # ce_loss = alpha_factor*ce_loss
            tg = target.argmax(1)
            alpha_factor = torch.ones(tg.size()) * self.weight
            if torch.cuda.is_available():
                alpha_factor = alpha_factor.cuda()
            alpha_factor = torch.where(torch.eq(tg, 0), alpha_factor, 1.0 - alpha_factor)
            logged_x_pred = torch.log(F.softmax(input, dim=1))
            ce_loss = -torch.sum(target * logged_x_pred, dim=1)
            ce_loss = alpha_factor * ce_loss
        else:
            logged_x_pred = torch.log(F.softmax(input, dim=1))
            ce_loss = -torch.sum(target * logged_x_pred, dim=1)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        # focal_loss = alpha_factor*focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SharpnessAwareMinimization(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, clip_norm=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SharpnessAwareMinimization, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.clip_norm = clip_norm
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False, clip_norm=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

            if clip_norm:
                torch.nn.utils.clip_grad_norm_(group["params"], 5)

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError("SAM doesn't work like the other optimizers, you should first call `first_step` and the `second_step`; see the documentation for more info.")

    def _grad_norm(self):
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm