#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from torch import einsum
import torch
import torch.nn.functional as F

from utils import simplex, sset


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)


class DiceLoss:
    def __init__(self, **kwargs):
        # self.idk: list[int] of classes to include in the loss
        self.idk = kwargs['idk']
        self.eps: float = kwargs['eps'] if 'eps' in kwargs else 1e-6
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax: torch.Tensor, weak_target: torch.Tensor) -> torch.Tensor:
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        p = pred_softmax[:, self.idk, ...]
        t = weak_target[:, self.idk, ...].float()

        # Compute per-class dice and then average
        dims = (0, 2, 3)
        intersection = (p * t).sum(dim=dims)
        cardinality = p.sum(dim=dims) + t.sum(dim=dims)
        dice_per_class = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        loss = 1.0 - dice_per_class.mean()
        return loss


class CEDiceLoss:
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, **kwargs):
        # Maintain the same idk convention and pass-through kwargs
        self.idk = kwargs['idk']
        self.alpha = alpha
        self.beta = beta
        self._ce = CrossEntropy(idk=self.idk)
        self._dice = DiceLoss(idk=self.idk)
        print(f"Initialized {self.__class__.__name__} with alpha={alpha}, beta={beta}, idk={self.idk}")

    def __call__(self, pred_softmax: torch.Tensor, weak_target: torch.Tensor) -> torch.Tensor:
        return self.alpha * self._ce(pred_softmax, weak_target) + self.beta * self._dice(pred_softmax, weak_target)


class FocalCrossEntropy:
    def __init__(self, gamma: float = 2.0, alpha: float | None = None, **kwargs):
        # self.idk: list[int] of classes to include
        self.idk = kwargs['idk']
        self.gamma = gamma
        self.alpha = alpha
        print(f"Initialized {self.__class__.__name__} with gamma={gamma}, alpha={alpha}, idk={self.idk}")

    def __call__(self, pred_softmax: torch.Tensor, weak_target: torch.Tensor) -> torch.Tensor:
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        p = torch.clamp(pred_softmax[:, self.idk, ...], min=1e-10, max=1.0)
        t = weak_target[:, self.idk, ...].float()

        # standard CE on probabilities: - y * log(p)
        ce = - (t * p.log())
        pt = (t * p).sum(dim=1, keepdim=True) + 1e-10  # p for the true class at each pixel
        modulator = (1.0 - pt) ** self.gamma
        focal = modulator * ce

        if self.alpha is not None:
            # support scalar alpha or per-class alpha list/tuple
            if isinstance(self.alpha, (list, tuple)):
                device = focal.device
                alpha_vec = torch.tensor(self.alpha, dtype=focal.dtype, device=device)
                alpha_map = alpha_vec[self.idk].view(1, -1, 1, 1)
                focal = alpha_map * focal
            else:
                focal = self.alpha * focal

        # average over considered classes/pixels
        mask = t.sum(dim=1, keepdim=True)  # 1 at labeled pixels among selected classes
        denom = mask.sum() + 1e-10
        loss = focal.sum() / denom
        return loss
