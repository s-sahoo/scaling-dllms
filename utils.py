"""Console logger utilities.

Copied from https://github.com/HazyResearch/transformers/blob/master/src/utils/utils.py
Copied from https://docs.python.org/3/howto/logging-cookbook.html#using-a-context-manager-for-selective-logging
"""

import logging
import json
import math
from typing import List

import fsspec
import lightning
import torch
from timm.scheduler import CosineLRScheduler


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
  """
  Global-step scheduler: linear warmup -> cosine decay.

  Args:
    optimizer: torch optimizer
    total_steps: total number of optimizer steps in the entire training run
    warmup_steps: linear warmup steps from 0 -> base lr
    min_lr: final learning rate at the end of decay (per group)
    last_epoch: DO NOT pass directly; Lightning manages stepping. Left for state restoration.

  Notes:
    - `last_epoch` here tracks *optimizer steps taken* (i.e., global steps within the scheduler).
    - Works with Lightning when you set scheduler dict {"interval": "step"}.
  """
  def __init__(
    self,
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 2000,
    min_lr: float = 0.0,
    last_epoch: int = -1):
    assert total_steps > 0, 'total_steps must be > 0'
    assert 0 < warmup_steps < total_steps, 'warmup_steps must be in (0, total_steps)'
    self.total_steps = total_steps
    self.warmup_steps = int(warmup_steps)
    self.min_lr = min_lr
    super().__init__(optimizer, last_epoch)

  def _lr_at(self, step: int, base_lr: float) -> float:
    # step is 0-based after super().__init__,
    # self.last_epoch starts at -1
    if step < self.warmup_steps:
        return base_lr * (step + 1) / self.warmup_steps
    progress = (step - self.warmup_steps) / (
      self.total_steps - self.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return self.min_lr + (base_lr - self.min_lr) * cosine

  def get_lr(self) -> List[float]:
    step = max(0, self.last_epoch)  # last_epoch counts calls to step()
    return [self._lr_at(step, base_lr=group['initial_lr'])
            for group in self.optimizer.param_groups]


def count_parameters(model):
  return sum(p.numel()
             for p in model.parameters()
             if p.requires_grad)

def fsspec_exists(filename):
  """Check if a file exists using fsspec."""
  fs, _ = fsspec.core.url_to_fs(filename)
  return fs.exists(filename)


def fsspec_listdir(dirname):
  """Listdir in manner compatible with fsspec."""
  fs, _ = fsspec.core.url_to_fs(dirname)
  return fs.ls(dirname)


def fsspec_mkdirs(dirname, exist_ok=True):
  """Mkdirs in manner compatible with fsspec."""
  fs, _ = fsspec.core.url_to_fs(dirname)
  fs.makedirs(dirname, exist_ok=exist_ok)


def print_nans(tensor, name):
  if torch.isnan(tensor).any():
    print(name, tensor)


def flops_to_max_steps(target_flops_1e18, global_batch_size,
                       json_path):
  with fsspec.open(json_path) as f:
    config = json.load(f)
  return int(target_flops_1e18 * 1e18
             / config['flops'] / global_batch_size / 3)


class LRHalveScheduler:
  def __init__(self, warmup_steps, n_halve_steps):
    self.warmup_steps = warmup_steps
    self.n_halve_steps = n_halve_steps
  
  def __call__(self, current_step):
    if current_step < self.warmup_steps:
      return current_step / self.warmup_steps
    return 0.5 ** ((current_step - self.warmup_steps)
                   // self.n_halve_steps)


class CosineDecayWarmupLRScheduler(
  CosineLRScheduler,
  torch.optim.lr_scheduler._LRScheduler):
  """Wrap timm.scheduler.CosineLRScheduler
  Enables calling scheduler.step() without passing in epoch.
  Supports resuming as well.
  Adapted from:
    https://github.com/HazyResearch/hyena-dna/blob/main/src/utils/optim/schedulers.py
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._last_epoch = -1
    self.step(epoch=0)

  def step(self, epoch=None):
    if epoch is None:
      self._last_epoch += 1
    else:
      self._last_epoch = epoch
    # We call either step or step_update, depending on
    # whether we're using the scheduler every epoch or every
    # step.
    # Otherwise, lightning will always call step (i.e.,
    # meant for each epoch), and if we set scheduler
    # interval to "step", then the learning rate update will
    # be wrong.
    if self.t_in_epochs:
      super().step(epoch=self._last_epoch)
    else:
      super().step_update(num_updates=self._last_epoch)


class LoggingContext:
  """Context manager for selective logging."""
  def __init__(self, logger, level=None, handler=None, close=True):
    self.logger = logger
    self.level = level
    self.handler = handler
    self.close = close

  def __enter__(self):
    if self.level is not None:
      self.old_level = self.logger.level
      self.logger.setLevel(self.level)
    if self.handler:
      self.logger.addHandler(self.handler)

  def __exit__(self, et, ev, tb):
    if self.level is not None:
      self.logger.setLevel(self.old_level)
    if self.handler:
      self.logger.removeHandler(self.handler)
    if self.handler and self.close:
      self.handler.close()


class GradientInspectionCallback(lightning.Callback):
    def __init__(self, num_grads_log):
        self.num_grads_log = 10

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
      gradients = []
      for name, param in pl_module.backbone.blocks.named_parameters():
          gradients.append(param.grad.view(-1))

      if gradients:
        grads = torch.cat((gradients))
        if not hasattr(pl_module, 'grad_accum_buffer'):
          pl_module.grad_step = torch.tensor(
            0, device=pl_module.device)
          pl_module.grad_accum_buffer = torch.zeros(
            self.num_grads_log,
            grads.shape[0],
            device=pl_module.device)
        pl_module.grad_accum_buffer[pl_module.grad_step] = grads
        pl_module.grad_step += 1

      if (hasattr(pl_module, 'grad_accum_buffer') 
          and pl_module.grad_step == self.num_grads_log):
        grads = pl_module.grad_accum_buffer
        grad_var = grads.std(0).mean()
        pl_module.log(name='trainer/grad_var',
                      value=grad_var.item(),
                      on_step=True,
                      on_epoch=False,
                      sync_dist=True)
        # import ipdb; ipdb.set_trace()
        # should save the grads tensor as a numpy array
        # and visualize mean, median, top-k
        pl_module.grad_accum_buffer.zero_()
        pl_module.grad_step = 0


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
  """Initializes multi-GPU-friendly python logger."""

  logger = logging.getLogger(name)
  logger.setLevel(level)

  # this ensures all logging levels get marked with the rank zero decorator
  # otherwise logs would get multiplied for each GPU process in multi-GPU setup
  for level in ('debug', 'info', 'warning', 'error',
                'exception', 'fatal', 'critical'):
    setattr(logger,
            level,
            lightning.pytorch.utilities.rank_zero_only(
              getattr(logger, level)))

  return logger


# Copied from https://github.com/jdeschena/sdtt/blob/bbc54d5b3c5fcffd79602cff17ed34dde1f3eff6/src/sdtt/core/sampling/utils.py#L10
def top_k_top_p_filtering(
    logits,
    top_k=0,
    top_p=0.0,
    filter_value=-float("Inf"),
    dim=-1):
    """Filter a distribution of logits using top-k/top-p (nucleus) filtering.
    Adapted from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317

    Args:
      logits (Tensor): Tensor of logits
      top_k (int, optional): Number of top values to keep.
          Deactivated if k is 0. Defaults to 0.
      top_p (float, optional): Cumulative mass to retain.
          Deactivated if p = 0. Defaults to 0.0.
      filter_value (float, optional): Fill value to replace
          the entries removed by top-k/top-p filtering.
          Defaults to -float('Inf').
      dim (int, optional): Dimension of the filtering. Defaults to -1.

    Returns:
        logits: Tensor whose axis `dim` was filtered.
    """
    if dim != -1:
      logits = torch.transpose(logits, dim, -1)

    assert top_k < logits.size(dim)
    if top_k > 0:
      # Remove all tokens with a probability less than
      # the last token of the top-k
      values, _ = torch.topk(logits, k=top_k, dim=-1)
      to_remove_mask = (
          logits < torch.min(values, dim=-1, keepdim=True)[0]
      )  # min returns a tuple (values, indices)
      logits[to_remove_mask] = filter_value

    if top_p > 0.0:
      sorted_logits, sorted_indices = torch.sort(
        logits, descending=True, dim=-1)
      cum_probs = torch.cumsum(
        torch.softmax(sorted_logits, dim=-1), dim=-1)

      sorted_indices_to_remove = cum_probs > top_p
      # Ensures at least one token is kept
      sorted_indices_to_remove[..., 1:] = \
        sorted_indices_to_remove[..., :-1].clone()
      sorted_indices_to_remove[..., 0] = 0

      mask_to_remove = torch.empty_like(sorted_indices_to_remove)
      mask_to_remove.scatter_(dim=-1,
                              index=sorted_indices,
                              src=sorted_indices_to_remove)
      logits[mask_to_remove] = filter_value

    if dim != -1:
      logits = torch.transpose(logits, dim, -1)

    return logits


def get_reverse_indices(indices):
  """
  indices: LongTensor of shape [B, N] representing permutations
  returns: LongTensor of shape [B, N] representing the inverse permutations
  """
  B, N = indices.shape
  reverse_indices = torch.empty_like(indices)
  arange = torch.arange(N, device=indices.device).unsqueeze(0).expand(B, -1)
  reverse_indices.scatter_(1, indices, arange)
  return reverse_indices

