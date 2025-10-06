#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec, Caroline Magg

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

import argparse
import warnings
import random
import os
from typing import Any
from pathlib import Path
from pprint import pprint
from operator import itemgetter
from shutil import copytree, rmtree

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
import wandb
from sam import SAM
from monai.optimizers import Novograd, WarmupCosineSchedule, generate_param_groups

# Ranger optimizer (RAdam + LookAhead + Gradient Centralization)
import torch_optimizer as optim

# CLMR (Cyclic Learning/Momentum Rate) for Nesterov SGD
from clmr import CreativeCLMRScheduler
# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print(">> Loaded environment variables from .env file")
except ImportError:
    print(">> python-dotenv not installed. Install with: pip install python-dotenv")
    print(">> Or set environment variables manually")

from functools import partial 
from copy import deepcopy

from dataset import SliceDataset
from ShallowNet import shallowCNN
from ENet import ENet
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   save_images)

from losses import (CrossEntropy)


class ModelEMA:
    """
    Exponential Moving Average of model parameters.
    Keeps a moving averaged copy of the model to improve generalization.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        # Shadow parameters on CPU in fp32
        self.shadow_params = {k: v.detach().cpu().float().clone() for k, v in model.state_dict().items()}
        self.backup_params = None

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.state_dict().items():
            if name in self.shadow_params:
                self.shadow_params[name].mul_(self.decay).add_(param.detach().cpu().float(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        self.backup_params = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow_params, strict=False)

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if self.backup_params is not None:
            model.load_state_dict(self.backup_params, strict=False)
            self.backup_params = None


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across different libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f">> Random seed set to {seed} for reproducibility")


datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the classes with C (often used for the number of Channel)
datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_CLEAN"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}

def img_transform(img):
        img = img.convert('L')
        img = np.array(img)[np.newaxis, ...]
        img = img / 255  # max <= 1
        img = torch.tensor(img, dtype=torch.float32)
        return img

def gt_transform(K, img):
        img = np.array(img)[...]
        # The idea is that the classes are mapped to {0, 255} for binary cases
        # {0, 85, 170, 255} for 4 classes
        # {0, 51, 102, 153, 204, 255} for 6 classes
        # Very sketchy but that works here and that simplifies visualization
        img = img / (255 / (K - 1)) if K != 5 else img / 63  # max <= 1
        img = torch.tensor(img, dtype=torch.int64)[None, ...]  # Add one dimension to simulate batch
        img = class2one_hot(img, K=K)
        return img[0]

def setup(args) -> tuple[nn.Module, Any, Any, Any, DataLoader, DataLoader, int]:
    # Initialize wandb
    wandb_mode = "offline" if args.wandb_offline else "online"
    
    # Check if wandb API key is available
    api_key = os.getenv('WANDB_API_KEY')
    if not api_key and wandb_mode == "online":
        print(">> Warning: WANDB_API_KEY not found in environment variables")
        print(">> Switching to offline mode. Set WANDB_API_KEY in .env file for online mode")
        wandb_mode = "offline"
    
    # Set experiment name
    if args.wandb_name:
        experiment_name = args.wandb_name
    else:
        experiment_name = f"{args.dataset}_{args.mode}_{args.epochs}epochs"
    
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=experiment_name,
        mode=wandb_mode,
        config={
            "dataset": args.dataset,
            "mode": args.mode,
            "epochs": args.epochs,
            "gpu": args.gpu,
            "debug": args.debug,
            "dest": str(args.dest),
            "seed": args.seed,
            "experiment_name": experiment_name
        }
    )
    
    # Networks and scheduler
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    K: int = datasets_params[args.dataset]['K']
    kernels: int = datasets_params[args.dataset]['kernels'] if 'kernels' in datasets_params[args.dataset] else 8
    factor: int = datasets_params[args.dataset]['factor'] if 'factor' in datasets_params[args.dataset] else 2
    net = datasets_params[args.dataset]['net'](1, K, kernels=kernels, factor=factor)
    net.init_weights()
    net.to(device)

    # Dataset part
    B: int = datasets_params[args.dataset]['B']
    root_dir = Path("data") / args.dataset
    
    train_set = SliceDataset('train',
                             root_dir,
                             img_transform=img_transform,
                             gt_transform= partial(gt_transform, K),
                             debug=args.debug)
    train_loader = DataLoader(train_set,
                              batch_size=B,
                              num_workers=5,
                              shuffle=True)

    val_set = SliceDataset('val',
                           root_dir,
                           img_transform=img_transform,
                           gt_transform=partial(gt_transform, K),
                           debug=args.debug)
    val_loader = DataLoader(val_set,
                            batch_size=B,
                            num_workers=5,
                            shuffle=False)
    
    # Initialize optimizer based on args with appropriate learning rates
    if args.optimizer == "adamw":
        lr = 0.0005
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999))
        optimizer_config = {
            "optimizer": "AdamW",
            "betas": (0.9, 0.999),
        }
    elif args.optimizer == "sam":
        lr = 0.01  # SAM with SGD needs higher LR than AdamW
        base_optimizer = torch.optim.SGD
        optimizer = SAM(net.parameters(), base_optimizer, lr=lr, momentum=0.9, rho=0.05)
        optimizer_config = {
            "optimizer": "SAM",
            "base_optimizer": "SGD",
            "momentum": 0.9,
            "rho": 0.05,
        }
    elif args.optimizer == "novograd":
        lr = 0.001  # Base learning rate for Novograd
        
        if args.use_layer_wise_lr:
            # Use layer-wise learning rates as recommended in MONAI docs
            # Different learning rates for encoder and decoder parts
            print(">> Using layer-wise learning rates for Novograd")
            
            # Create parameter groups with different learning rates
            # For ENet: initial layers, encoder layers, decoder layers
            param_groups = generate_param_groups(
                network=net,
                layer_matches=[
                    # Initial layers: conv0, maxpool0, bottleneck1_0
                    lambda x: nn.ModuleList([x.conv0, x.maxpool0, x.bottleneck1_0]) if hasattr(x, 'conv0') else nn.ModuleList(),
                    # Encoder layers: bottleneck1_1, bottleneck2_0, bottleneck2_1, bottleneck3
                    lambda x: nn.ModuleList([x.bottleneck1_1, x.bottleneck2_0, x.bottleneck2_1, x.bottleneck3]) if hasattr(x, 'bottleneck1_1') else nn.ModuleList(),
                ],
                match_types=["select", "select"],
                lr_values=[lr * 0.5, lr * 0.75],  # Lower LR for early layers
                include_others=True  # Include decoder/output with base LR
            )
            
            optimizer = Novograd(
                param_groups,
                lr=lr,  # Base LR for remaining layers
                betas=(0.9, 0.98),
                weight_decay=0.001,
                grad_averaging=False,
                amsgrad=False
            )
            optimizer_config = {
                "optimizer": "Novograd",
                "betas": (0.9, 0.98),
                "weight_decay": 0.001,
                "grad_averaging": False,
                "amsgrad": False,
                "layer_wise_lr": True,
                "lr_initial": lr * 0.5,
                "lr_encoder": lr * 0.75,
                "lr_decoder": lr,  # Base LR for decoder/remaining layers
            }
        else:
            # Simple Novograd configuration without layer-wise LR
            optimizer = Novograd(
                net.parameters(),
                lr=lr,
                betas=(0.9, 0.98),
                weight_decay=0.001,
                grad_averaging=False,
                amsgrad=False
            )
            optimizer_config = {
                "optimizer": "Novograd",
                "betas": (0.9, 0.98),
                "weight_decay": 0.001,
                "grad_averaging": False,
                "amsgrad": False,
                "layer_wise_lr": False,
            }
    elif args.optimizer == "ranger":
        lr = 0.005  

        optimizer = optim.Ranger(
            net.parameters(),
            lr=lr,  # More frequent lookahead updates for better adaptation
            weight_decay=0.01,
            alpha=0.8,
            k=5,
            betas=(0.95, 0.999),
            eps=1e-8,

        )
        optimizer_config = {
            "optimizer": "Ranger",
            "weight_decay": 0.01,
            "alpha": 0.8,
            "k": 5,
            "betas": (0.95, 0.999),
            "eps": 1e-8,
        }
    elif args.optimizer == "clmr":
        # CreativeCLMR: Enhanced Cyclic Learning/Momentum Rate with Nesterov SGD
        # Creative combination of multiple advanced techniques to beat baseline across ALL metrics

        # Optimized LR range for SEGTHOR (based on empirical testing)
        lr_min = 2e-5   # Conservative minimum for stability
        lr_max = 8e-4   # Maximum that works well with SEGTHOR data

        # Optimized momentum range (narrower for better stability)
        mom_min = 0.88  # Higher minimum for better convergence
        mom_max = 0.95  # Lower maximum for less oscillation

        # Create Nesterov SGD optimizer with gradient centralization
        lr = lr_min  # Start conservative (will ramp up)

        # Enhanced SGD parameters for better performance
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=lr,
            momentum=mom_min,  # Start with lower momentum
            nesterov=True,     # Required for CLMR
            weight_decay=5e-5  # Lighter regularization
        )

        # Add gradient centralization for better training stability
        # This helps with all metrics by improving gradient flow
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.requires_grad:
                    param.register_hook(lambda grad: grad - grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))

        # Enhanced cycle configuration
        steps_per_epoch = len(train_loader)
        base_cycle_steps = steps_per_epoch * 2  # 2 epochs base cycle

        # Use enhanced CLMR scheduler
        scheduler = CreativeCLMRScheduler(
            optimizer,
            lr_min=lr_min,
            lr_max=lr_max,
            base_cycle_steps=base_cycle_steps,
            mom_min=mom_min,
            mom_max=mom_max,
            mom_cycle_steps=base_cycle_steps,  # Same as LR cycle
            antiphase=True,   # Anti-phase: better exploration
            adaptive_cycles=True,  # Adaptive cycle length
            lookahead_steps=5,  # Lookahead mechanism for better convergence
            gradient_centralization=True,  # Better gradient flow
            layer_wise_lr=True,  # Different LRs for different layers
            momentum_reset_interval=1000,  # Reset momentum periodically
        )

        optimizer_config = {
            "optimizer": "CreativeCLMR+NesterovSGD",
            "lr_min": lr_min,
            "lr_max": lr_max,
            "mom_min": mom_min,
            "mom_max": mom_max,
            "base_cycle_steps": base_cycle_steps,
            "cycle_epochs": 2,  # 2 epochs per base cycle
            "antiphase": True,
            "adaptive_cycles": True,
            "lookahead_steps": 5,
            "gradient_centralization": True,
            "layer_wise_lr": True,
            "momentum_reset_interval": 1000,
            "weight_decay": 5e-5,
        }
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # LearningRateFinder removed per project simplification
    
    # Create learning rate scheduler (compatible with all optimizers)
    # Note: CLMR has its own scheduler, so we skip WarmupCosineSchedule for it
    if args.optimizer != "clmr":
        scheduler = None
        if args.use_scheduler:
            total_steps = args.epochs * len(train_loader)
            # Ranger benefits from longer warmup
            warmup_pct = 0.15 if args.optimizer == "ranger" else 0.1
            warmup_steps = int(warmup_pct * total_steps)
            # Less aggressive decay for Ranger to maintain exploration
            end_lr = lr * 0.05 if args.optimizer == "ranger" else lr * 0.01
            
            scheduler = WarmupCosineSchedule(
                optimizer,
                warmup_steps=warmup_steps,
                t_total=total_steps,
                end_lr=end_lr,
                cycles=0.5,
                last_epoch=-1,
                warmup_multiplier=0  # Start warmup from 0
            )
            print(f">> Using WarmupCosineSchedule: warmup_steps={warmup_steps} ({warmup_pct*100:.0f}%), "
                  f"total_steps={total_steps}, end_lr={end_lr:.2e}")
    else:
        # CLMR already created its scheduler above
        print(f">> Using CreativeCLMR: lr=[{lr_min:.2e}, {lr_max:.2e}], "
              f"momentum=[{mom_min:.2f}, {mom_max:.2f}], cycle={base_cycle_steps//steps_per_epoch}ep, antiphase=True, adaptive")
    
    # Log model architecture and hyperparameters to wandb
    wandb_config = {
        "learning_rate": lr,
        **optimizer_config,
        "num_classes": K,
        "kernels": kernels,
        "factor": factor,
        "batch_size": B,
        "num_workers": 5,
        "seed": args.seed,
        "use_scheduler": args.use_scheduler,
        # LR finder disabled
        "use_layer_wise_lr": args.use_layer_wise_lr,
        "grad_clip": args.grad_clip,
    }
    
    # Log scheduler-specific config
    if args.optimizer == "clmr":
        wandb_config["scheduler"] = "CLMR"
    elif scheduler is not None and args.use_scheduler:
        wandb_config["scheduler"] = "WarmupCosineSchedule"
        wandb_config["warmup_steps"] = warmup_steps
        wandb_config["total_steps"] = total_steps
        wandb_config["end_lr"] = end_lr
        wandb_config["warmup_pct"] = warmup_pct
    
    # Log Ranger-specific grad clip multiplier
    if args.optimizer == "ranger":
        wandb_config["ranger_grad_clip_multiplier"] = 2.0
    
    wandb.config.update(wandb_config)
    
    # Log model architecture (commented out due to pickle issues with wandb.watch)
    # wandb.watch(net, log="all", log_freq=10)

    args.dest.mkdir(parents=True, exist_ok=True)

    return (net, optimizer, scheduler, device, train_loader, val_loader, K)


def runTraining(args):
    print(f">>> Setting up to train on {args.dataset} with {args.mode} using {args.optimizer} optimizer")
    net, optimizer, scheduler, device, train_loader, val_loader, K = setup(args)
    # EMA for better generalization (helps metrics beyond Dice)
    ema = ModelEMA(net, decay=0.999)

    if args.mode == "full":
        loss_fn = CrossEntropy(idk=list(range(K)))  # Supervise both background and foreground
    elif args.mode in ["partial"] and args.dataset == 'SEGTHOR':
        loss_fn = CrossEntropy(idk=[0, 1, 3, 4])  # Do not supervise the heart (class 2)
    else:
        raise ValueError(args.mode, args.dataset)

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

    best_dice: float = 0

    for e in range(args.epochs):
        for m in ['train', 'val']:
            if m == 'train':
                net.train()
                opt = optimizer
                cm = Dcm
                desc = f">> Training   ({e: 4d})"
                loader = train_loader
                log_loss = log_loss_tra
                log_dice = log_dice_tra
            elif m == 'val':
                net.eval()
                opt = None
                cm = torch.no_grad
                desc = f">> Validation ({e: 4d})"
                loader = val_loader
                log_loss = log_loss_val
                log_dice = log_dice_val

            with cm():  # Either dummy context manager, or the torch.no_grad for validation
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data['images'].to(device)
                    gt = data['gts'].to(device)

                    if opt:  # So only for training
                        opt.zero_grad()

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    B, _, W, H = img.shape

                    pred_logits = net(img)
                    pred_probs = F.softmax(1 * pred_logits, dim=1)  # 1 is the temperature parameter

                    # Metrics computation, not used for training
                    pred_seg = probs2one_hot(pred_probs)
                    log_dice[e, j:j + B, :] = dice_coef(pred_seg, gt)  # One DSC value per sample and per class

                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = loss.item()  # One loss value per batch (averaged in the loss)

                    if opt:  # Only for training
                        loss.backward()
                        
                        # Gradient clipping for stability - optimizer-specific thresholds
                        if args.grad_clip > 0:
                            # Ranger uses GC for regularization, needs less aggressive clipping
                            if opt == optim.Ranger:
                                clip_value = args.grad_clip * 2.0
                            else:
                                clip_value = args.grad_clip
                            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_value)
                        
                        # SAM optimizer requires two-step optimization
                        if isinstance(opt, SAM):
                            opt.first_step(zero_grad=True)
                            
                            # Second forward-backward pass
                            pred_logits_2 = net(img)
                            pred_probs_2 = F.softmax(1 * pred_logits_2, dim=1)
                            loss_2 = loss_fn(pred_probs_2, gt)
                            loss_2.backward()
                            
                            # Gradient clipping for second pass too
                            if args.grad_clip > 0:
                                clip_value = args.grad_clip * 2.0 if opt == optim.Ranger else args.grad_clip
                                torch.nn.utils.clip_grad_norm_(net.parameters(), clip_value)
                            
                            opt.second_step(zero_grad=True)
                        else:
                            opt.step()
                            # Update EMA after optimizer step
                            ema.update(net)
                        
                        # Update learning rate scheduler if present
                        if scheduler is not None:
                            scheduler.step()

                    if m == 'val':
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning)
                            predicted_class: Tensor = probs2class(pred_probs)
                            mult: int = 63 if K == 5 else (255 / (K - 1))
                            save_images(predicted_class * mult,
                                        data['stems'],
                                        args.dest / f"iter{e:03d}" / m)

                    j += B  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    postfix_dict: dict[str, str] = {"Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                                                    "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    if K > 2:
                        postfix_dict |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                                         for k in range(1, K)}
                    tq_iter.set_postfix(postfix_dict)

        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)

        # Log metrics to wandb
        train_loss_epoch = log_loss_tra[e, :].mean().item()
        val_loss_epoch = log_loss_val[e, :].mean().item()
        train_dice_epoch = log_dice_tra[e, :, 1:].mean().item()  # Exclude background class
        val_dice_epoch = log_dice_val[e, :, 1:].mean().item()   # Exclude background class

        # Evaluate EMA on validation for better generalization metrics
        ema.store(net)
        ema.copy_to(net)
        net.eval()
        with torch.no_grad():
            j_eval = 0
            for data in val_loader:
                img = data['images'].to(device)
                gt = data['gts'].to(device)
                pred_logits = net(img)
                pred_probs = F.softmax(1 * pred_logits, dim=1)
                pred_seg = probs2one_hot(pred_probs)
                log_dice_val[e, j_eval:j_eval + img.shape[0], :] = dice_coef(pred_seg, gt)
                j_eval += img.shape[0]
        ema.restore(net)
        
        # Get current learning rate and momentum (for CLMR)
        current_lr = optimizer.param_groups[0]['lr']
        
    # Log per-class dice scores
        wandb_log = {
            "epoch": e,
            "train_loss": train_loss_epoch,
            "val_loss": val_loss_epoch,
            "train_dice": train_dice_epoch,
            "val_dice": val_dice_epoch,
            "learning_rate": current_lr
        }
        
        # Log momentum and effective cycle position for CLMR
        if args.optimizer == "clmr":
            current_momentum = optimizer.param_groups[0].get('momentum', None)
            if current_momentum is not None:
                wandb_log["momentum"] = current_momentum
        # expose LR bounds for debugging (copy from wandb config which already contains them)
        wandb_log["clmr_lr_min"] = wandb.config.get("lr_min")
        wandb_log["clmr_lr_max"] = wandb.config.get("lr_max")
        
        # Add per-class dice scores
        for k in range(1, K):  # Skip background class
            wandb_log[f"train_dice_class_{k}"] = log_dice_tra[e, :, k].mean().item()
            wandb_log[f"val_dice_class_{k}"] = log_dice_val[e, :, k].mean().item()
        
        wandb.log(wandb_log, step=e)

        current_dice: float = log_dice_val[e, :, 1:].mean().item()
        if current_dice > best_dice:
            message = f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC"
            print(message)
            best_dice = current_dice
            
            # Log best dice improvement to wandb
            wandb.log({"best_dice": current_dice, "best_epoch": e}, step=e)
            
            with open(args.dest / "best_epoch.txt", 'w') as f:
                f.write(message)

            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                rmtree(best_folder)
            copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")
            
            # Save model artifacts to wandb
            wandb.save(str(args.dest / "bestmodel.pkl"))
            wandb.save(str(args.dest / "bestweights.pt"))
    
    # Final logging and artifact saving
    print(f">>> Training completed. Best dice: {best_dice:.3f}")
    wandb.log({"final_best_dice": best_dice}, step=args.epochs-1)
    
    # Create wandb artifacts for the complete experiment
    artifact = wandb.Artifact(
        name=f"model_{args.dataset}_{args.mode}",
        type="model",
        description=f"Best model for {args.dataset} dataset in {args.mode} mode"
    )
    artifact.add_file(str(args.dest / "bestweights.pt"))
    artifact.add_file(str(args.dest / "bestmodel.pkl"))
    artifact.add_file(str(args.dest / "best_epoch.txt"))
    wandb.log_artifact(artifact)
    
    # Create metrics artifact
    metrics_artifact = wandb.Artifact(
        name=f"metrics_{args.dataset}_{args.mode}",
        type="metrics",
        description=f"Training metrics for {args.dataset} dataset in {args.mode} mode"
    )
    metrics_artifact.add_file(str(args.dest / "loss_tra.npy"))
    metrics_artifact.add_file(str(args.dest / "dice_tra.npy"))
    metrics_artifact.add_file(str(args.dest / "loss_val.npy"))
    metrics_artifact.add_file(str(args.dest / "dice_val.npy"))
    wandb.log_artifact(metrics_artifact)
    
    # Finish wandb run
    wandb.finish()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--dataset', default='TOY2', choices=datasets_params.keys())
    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the results (predictions and weights).")

    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logics around epochs and logging easily.")
    parser.add_argument('--wandb_project', type=str, default='ai4mi-segthor',
                        help="Wandb project name")
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help="Wandb entity name (optional)")
    parser.add_argument('--wandb_offline', action='store_true',
                        help="Run wandb in offline mode")
    parser.add_argument('--wandb_name', type=str, default=None,
                        help="Custom name for wandb experiment run")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument('--optimizer', type=str, default='novograd', choices=['adamw', 'sam', 'novograd', 'ranger', 'clmr'],
                        help="Optimizer to use for training (default: novograd)")
    parser.add_argument('--use_scheduler', action='store_true', default=True,
                        help="Use WarmupCosineSchedule learning rate scheduler")
    parser.add_argument('--use_layer_wise_lr', action='store_true', default=False,
                        help="Use layer-wise learning rates (different LRs for encoder/decoder) - only for Novograd")
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help="Gradient clipping max norm (0 to disable, default: 1.0, Ranger uses 2x)")

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_random_seed(args.seed)

    pprint(args)

    runTraining(args)


if __name__ == '__main__':
    main()
