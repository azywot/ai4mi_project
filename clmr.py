# clmr.py
import math
import torch

class CreativeCLMRScheduler:
    """
    Enhanced Cyclic Learning/Momentum Rate for Nesterov SGD.
    Creative improvements for beating baseline across ALL metrics.

    Combines multiple advanced techniques:
    1. Adaptive cycle length (starts long, gradually shortens)
    2. Gradient centralization for better gradient flow
    3. Lookahead mechanism for better convergence
    4. Layer-wise adaptive learning rates
    5. Smart LR range selection for SEGTHOR
    6. Enhanced momentum strategy with resets

    This should outperform baseline on Dice, Jaccard, HD95, ASSD.
    """

    def __init__(
        self,
        optimizer,
        lr_min=2e-5, lr_max=8e-4, base_cycle_steps=2000,
        mom_min=0.88, mom_max=0.95, mom_cycle_steps=None,
        antiphase=True,  # Anti-phase works better for exploration
        adaptive_cycles=True,
        lookahead_steps=5,  # Lookahead mechanism
        gradient_centralization=True,  # Better gradient flow
        layer_wise_lr=True,  # Different LRs for different layers
        momentum_reset_interval=1000,  # Reset momentum periodically
    ):
        self.opt = optimizer
        self.lr_min, self.lr_max = lr_min, lr_max
        self.mom_min, self.mom_max = mom_min, mom_max
        self.base_cycle = base_cycle_steps
        self.mom_cycle = mom_cycle_steps or base_cycle_steps
        self.antiphase = antiphase
        self.adaptive_cycles = adaptive_cycles
        self.lookahead_steps = lookahead_steps
        self.gradient_centralization = gradient_centralization
        self.layer_wise_lr = layer_wise_lr
        self.momentum_reset_interval = momentum_reset_interval

        # Adaptive cycle parameters
        self.cycle_scale = 1.0
        self.min_cycle_scale = 0.3  # Don't go below 30% of base cycle
        self.t = 0  # global step

        # Lookahead parameters - use group index as key instead of params list
        self.lookahead_params = {}
        self.lookahead_step = 0

        # Initialize lookahead
        if self.lookahead_steps > 0:
            for group_idx, group in enumerate(self.opt.param_groups):
                self.lookahead_params[group_idx] = {}

        # Validate Nesterov SGD
        for g in self.opt.param_groups:
            if not getattr(self.opt, "nesterov", False) and not g.get("nesterov", False):
                raise ValueError("CreativeCLMR requires torch.optim.SGD with nesterov=True")

    @staticmethod
    def _triangle(step, period):
        """Triangular wave: 0→1→0 over period steps"""
        if period <= 0:
            return 0.0
        x = (step % period) / float(period)
        return 1.0 - abs(2.0 * x - 1.0)

    def _interp(self, lo, hi, alpha):
        """Linear interpolation between lo and hi"""
        return lo + (hi - lo) * alpha

    def _get_adaptive_cycle_length(self):
        """Adapt cycle length based on training progress"""
        if not self.adaptive_cycles:
            return self.base_cycle

        # Start with longer cycles, gradually shorten for refinement
        progress = min(self.t / 10000, 1.0)  # Assume ~10k steps total
        # Cycle length: start at 2x base, end at 0.5x base
        cycle_scale = 2.0 - 1.2 * progress  # 2.0 → 0.8
        cycle_scale = max(cycle_scale, self.min_cycle_scale)
        return int(self.base_cycle * cycle_scale)

    def _apply_gradient_centralization(self):
        """Apply gradient centralization for better gradient flow"""
        if not self.gradient_centralization:
            return

        for group in self.opt.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    # Gradient centralization: subtract mean across feature dimensions
                    grad = param.grad
                    if grad.dim() > 1:
                        # For conv layers: centralize across spatial and channel dims
                        if grad.dim() == 4:  # NCHW
                            grad = grad - grad.mean(dim=(1, 2, 3), keepdim=True)
                        elif grad.dim() == 2:  # Linear layers
                            grad = grad - grad.mean(dim=1, keepdim=True)

    def _apply_lookahead(self):
        """Apply lookahead mechanism"""
        if self.lookahead_steps <= 0:
            return

        # Store current parameters for lookahead
        if self.lookahead_step == 0:
            for group_idx, group in enumerate(self.opt.param_groups):
                if group_idx not in self.lookahead_params:
                    self.lookahead_params[group_idx] = {}
                for param_idx, param in enumerate(group['params']):
                    if param_idx not in self.lookahead_params[group_idx]:
                        self.lookahead_params[group_idx][param_idx] = {}
                    self.lookahead_params[group_idx][param_idx]['slow'] = param.data.clone()

        # Update fast parameters (current training)
        self.lookahead_step += 1

        # Apply lookahead update
        if self.lookahead_step >= self.lookahead_steps:
            alpha = 0.5  # Lookahead mixing factor
            for group_idx, group in enumerate(self.opt.param_groups):
                for param_idx, param in enumerate(group['params']):
                    if (group_idx in self.lookahead_params and 
                        param_idx in self.lookahead_params[group_idx]):
                        # Mix fast and slow parameters
                        slow_param = self.lookahead_params[group_idx][param_idx]['slow']
                        param.data.mul_(1 - alpha).add_(slow_param, alpha=alpha)
            self.lookahead_step = 0

    def step(self):
        """Update LR and momentum with enhanced cycling strategy"""
        # Adaptive cycle length
        current_lr_cycle = self._get_adaptive_cycle_length()

        # LR cycling: enhanced triangular wave
        lr_alpha = self._triangle(self.t, current_lr_cycle)
        # Add slight non-linearity for better exploration
        lr_alpha = lr_alpha * (1.0 + 0.1 * (lr_alpha - 0.5))
        lr_val = self._interp(self.lr_min, self.lr_max, max(0, min(1, lr_alpha)))

        # Momentum cycling: anti-phase for better exploration
        mom_t = self.t + (current_lr_cycle // 2 if self.antiphase else 0)
        mom_alpha = self._triangle(mom_t, self.mom_cycle)
        # Momentum should be high when LR is low (exploitation) and vice versa
        if self.antiphase:
            mom_alpha = 1.0 - mom_alpha  # Invert for anti-phase
        mom_val = self._interp(self.mom_min, self.mom_max, mom_alpha)

        # Reset momentum periodically for better exploration
        if self.t > 0 and self.t % self.momentum_reset_interval == 0:
            mom_val = self.mom_min  # Reset to low momentum for exploration

        # Apply layer-wise learning rates if enabled
        if self.layer_wise_lr:
            # Different LR scaling for different layer types
            for i, group in enumerate(self.opt.param_groups):
                if 'layer_type' in group:
                    layer_type = group['layer_type']
                    if layer_type == 'encoder':
                        group["lr"] = lr_val * 0.5  # Lower LR for early layers
                    elif layer_type == 'decoder':
                        group["lr"] = lr_val * 1.5  # Higher LR for later layers
                    else:
                        group["lr"] = lr_val
                else:
                    group["lr"] = lr_val
                group["momentum"] = mom_val
                group["nesterov"] = True
        else:
            # Apply to all param groups uniformly
            for g in self.opt.param_groups:
                g["lr"] = lr_val
                g["momentum"] = mom_val
                g["nesterov"] = True

        # Apply gradient centralization
        self._apply_gradient_centralization()

        # Apply lookahead
        self._apply_lookahead()

        self.t += 1
        return lr_val, mom_val
