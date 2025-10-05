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
from monai.losses import DiceLoss as MonaiDiceLoss
from monai.losses import GeneralizedDiceLoss as MonaiGDL
from monai.losses import TverskyLoss as MonaiTversky
from monai.losses import DiceCELoss as MonaiDiceCE
from monai.losses import FocalLoss as MonaiFocal
from monai.losses import HausdorffDTLoss as MonaiHausdorffDT
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


class GeneralizedDiceLoss:
    def __init__(self, **kwargs):
        # self.idk: list[int] of classes to include in the loss
        self.idk = kwargs['idk']
        self.eps: float = kwargs['eps'] if 'eps' in kwargs else 1e-6
        self.weight_type: str = kwargs['w_type'] if 'w_type' in kwargs else 'square'  # 'square' or 'simple'
        print(f"Initialized {self.__class__.__name__} with idk={self.idk}, w_type={self.weight_type}")

    def __call__(self, pred_softmax: torch.Tensor, weak_target: torch.Tensor) -> torch.Tensor:
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        p = pred_softmax[:, self.idk, ...]
        t = weak_target[:, self.idk, ...].float()

        dims = (0, 2, 3)
        t_sum = t.sum(dim=dims) + self.eps

        if self.weight_type == 'square':
            w = 1.0 / (t_sum * t_sum)
        else:
            w = 1.0 / t_sum

        intersect = (p * t).sum(dim=dims)
        p_sum = p.sum(dim=dims)

        num = 2.0 * (w * intersect).sum()
        den = (w * (p_sum + t_sum - self.eps)).sum() + self.eps
        loss = 1.0 - num / den
        return loss


class TverskyLoss:
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, **kwargs):
        self.idk = kwargs['idk']
        self.alpha = alpha
        self.beta = beta
        self.eps: float = kwargs['eps'] if 'eps' in kwargs else 1e-6
        print(f"Initialized {self.__class__.__name__} with alpha={alpha}, beta={beta}, idk={self.idk}")

    def __call__(self, pred_softmax: torch.Tensor, weak_target: torch.Tensor) -> torch.Tensor:
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        p = pred_softmax[:, self.idk, ...]
        t = weak_target[:, self.idk, ...].float()

        dims = (0, 2, 3)
        tp = (p * t).sum(dim=dims)
        fp = (p * (1.0 - t)).sum(dim=dims)
        fn = ((1.0 - p) * t).sum(dim=dims)

        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        loss = 1.0 - tversky.mean()
        return loss


class DiceFocalLoss:
    def __init__(self, alpha_dice: float = 0.5, beta_focal: float = 0.5, gamma: float = 2.0, **kwargs):
        self.idk = kwargs['idk']
        self.alpha_dice = alpha_dice
        self.beta_focal = beta_focal
        self.gamma = gamma
        self._dice = DiceLoss(idk=self.idk)
        self._focal = FocalCrossEntropy(idk=self.idk, gamma=gamma)
        print(f"Initialized {self.__class__.__name__} with alpha_dice={alpha_dice}, beta_focal={beta_focal}, gamma={gamma}, idk={self.idk}")

    def __call__(self, pred_softmax: torch.Tensor, weak_target: torch.Tensor) -> torch.Tensor:
        return self.alpha_dice * self._dice(pred_softmax, weak_target) + self.beta_focal * self._focal(pred_softmax, weak_target)


# Aliases for convenience / MONAI-like names
class DiceCELoss(CEDiceLoss):
    pass


class FocalLoss(FocalCrossEntropy):
    pass


# ------------------ MONAI wrappers ------------------
class MonaiDiceLossWrapper:
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        # include_background depends on whether 0 is in idk
        include_background = 0 in self.idk
        self._loss = MonaiDiceLoss(include_background=include_background, softmax=False, sigmoid=False, to_onehot_y=False)
        print(f"Initialized {self.__class__.__name__} with idk={self.idk}, include_background={include_background}")

    def __call__(self, pred_softmax: torch.Tensor, weak_target: torch.Tensor) -> torch.Tensor:
        # Mask classes per idk to mimic project behavior
        p = pred_softmax[:, self.idk, ...]
        t = weak_target[:, self.idk, ...].float()
        return self._loss(p, t)


class MonaiGeneralizedDiceLossWrapper:
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        include_background = 0 in self.idk
        self._loss = MonaiGDL(include_background=include_background, softmax=False, to_onehot_y=False)
        print(f"Initialized {self.__class__.__name__} with idk={self.idk}, include_background={include_background}")

    def __call__(self, pred_softmax: torch.Tensor, weak_target: torch.Tensor) -> torch.Tensor:
        p = pred_softmax[:, self.idk, ...]
        t = weak_target[:, self.idk, ...].float()
        return self._loss(p, t)


class MonaiTverskyLossWrapper:
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, **kwargs):
        self.idk = kwargs['idk']
        include_background = 0 in self.idk
        self._loss = MonaiTversky(include_background=include_background, alpha=alpha, beta=beta, softmax=False, to_onehot_y=False)
        print(f"Initialized {self.__class__.__name__} with idk={self.idk}, include_background={include_background}, alpha={alpha}, beta={beta}")

    def __call__(self, pred_softmax: torch.Tensor, weak_target: torch.Tensor) -> torch.Tensor:
        p = pred_softmax[:, self.idk, ...]
        t = weak_target[:, self.idk, ...].float()
        return self._loss(p, t)


class MonaiDiceCELossWrapper:
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        include_background = 0 in self.idk
        self._loss = MonaiDiceCE(include_background=include_background, softmax=False, to_onehot_y=False)
        print(f"Initialized {self.__class__.__name__} with idk={self.idk}, include_background={include_background}")

    def __call__(self, pred_softmax: torch.Tensor, weak_target: torch.Tensor) -> torch.Tensor:
        p = pred_softmax[:, self.idk, ...]
        t = weak_target[:, self.idk, ...].float()
        return self._loss(p, t)


class MonaiFocalLossWrapper:
    def __init__(self, gamma: float = 2.0, **kwargs):
        self.idk = kwargs['idk']
        self.gamma = gamma
        self._loss = MonaiFocal(gamma=gamma)
        print(f"Initialized {self.__class__.__name__} with idk={self.idk}, gamma={gamma}")

    def __call__(self, pred_softmax: torch.Tensor, weak_target: torch.Tensor) -> torch.Tensor:
        # MONAI Focal expects logits or probabilities? Its FocalLoss supports logits; we have probabilities.
        # We'll compute on probabilities by converting to per-pixel class index and one-hot remains; Monai FocalLoss expects logits by default.
        # To stay within our pipeline, convert probabilities to logits safely.
        eps = 1e-10
        logits = torch.log(torch.clamp(pred_softmax, min=eps))  # log-probs as logits surrogate
        p = logits[:, self.idk, ...]
        t = weak_target[:, self.idk, ...].float()
        return self._loss(p, t)


class MonaiHausdorffDTLossWrapper:
    def __init__(self, alpha: float = 2.0, include_background: bool | None = None, **kwargs):
        # include_background default: infer from idk if not explicitly provided
        self.idk = kwargs['idk']
        if include_background is None:
            include_background = 0 in self.idk
        # MONAI HausdorffDTLoss expects logits by default, but accepts probs if sigmoid/softmax is False
        # We'll pass probabilities directly, matching other wrappers, after masking channels.
        # alpha controls the distance transform power; defaults to 2.0 in MONAI
        self._loss = MonaiHausdorffDT(include_background=include_background, alpha=alpha)
        print(f"Initialized {self.__class__.__name__} with idk={self.idk}, include_background={include_background}, alpha={alpha}")

    def __call__(self, pred_softmax: torch.Tensor, weak_target: torch.Tensor) -> torch.Tensor:
        p = pred_softmax[:, self.idk, ...]
        t = weak_target[:, self.idk, ...].float()
        return self._loss(p, t)
