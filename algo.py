import functools
import collections

import numpy as np
import torch
import torch.nn.functional as F

import trainer_base
import utils


def _block_sampler_config(sub_context_length,
                          num_parallel_decode,
                          num_samples=1,
                          device='cuda'):
  sort_idx = torch.rand(
    num_samples, sub_context_length).argsort(
      descending=False).to(device)
  decoder_order = torch.arange(
      sub_context_length * num_parallel_decode).reshape(
          num_parallel_decode, sub_context_length).to(device)
  decode_order = torch.vstack(
      [decoder_order[:, s].T.reshape(-1) for s in sort_idx])
  unmask_k_tokens = [num_parallel_decode] * sub_context_length
  return unmask_k_tokens, decode_order


@functools.lru_cache
def _block_autoregressive_config(sub_context_length,
                                 sar_steps,
                                 num_parallel_decode,
                                 num_samples,
                                 device='cuda'):
  unmask_k_tokens = []
  anchors = []
  anchors = torch.arange(
    sar_steps * num_parallel_decode) * sub_context_length
  anchors = anchors.chunk(sar_steps)
  sort_idx = []
  for sar_block_idx in range(sar_steps):
    unmask_k_tokens.extend(
      [1] * num_parallel_decode
      + [num_parallel_decode] * (sub_context_length - 1))
    sort_idx.extend(
      [anchors[sar_block_idx] + i
        for i in range(sub_context_length)])
  sort_idx = torch.cat(sort_idx).repeat(num_samples, 1).to(device)
  return unmask_k_tokens, sort_idx


class AR(trainer_base.TrainerBase):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self.save_hyperparameters()
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    assert not self.config.algo.time_conditioning
    assert self.config.prior.type == 'none'

  def _process_model_input(self, x0, valid_tokens):
    input_tokens = x0[:, :-1]
    output_tokens = x0[:, 1:]
    valid_tokens = valid_tokens[:, 1:]
    return input_tokens, output_tokens, valid_tokens

  def nll(self, input_tokens, output_tokens,
          current_accumulation_step, train_mode):
    del train_mode, current_accumulation_step
    dummy_t0 = torch.zeros(input_tokens.shape[0],
                           dtype=self.dtype,
                           device=self.device)
    output = self.backbone(input_tokens, dummy_t0)
    output[:, :, self.mask_index] = self.neg_infinity
    output = output.log_softmax(-1)
    return - output.gather(
      -1, output_tokens[:, :, None])[:, :, 0]

  @torch.no_grad()
  def generate_samples(self, num_samples, **kwargs):
    # precompute token buffer
    num_pred_tokens = self.num_tokens - 1
    x = torch.zeros(
      (num_samples, num_pred_tokens + 1),
      dtype=torch.long,
      device=self.device)
    x[:, 0] = self.tokenizer.bos_token_id
    # precompute noise
    noise = (torch.distributions.Gumbel(0, 1)
             .sample((num_samples, num_pred_tokens, self.vocab_size))
             .to(self.device))
    if self.config.sampling.use_float64:
      noise = noise.to(torch.float64)
    kv_cache = self.config.sampling.kv_cache
    self.backbone.reset_kv_cache()
    sigma = torch.zeros(num_samples,
                        dtype=self.dtype,
                        device=self.device)
    for i in range(num_pred_tokens):
      output = self.backbone(
        x[:, :i + 1], sigma=sigma, x0=None, kv_cache=kv_cache)
      output[:, :, self.mask_index] += self.neg_infinity
      y = (output[:, -1, :] + noise[:, i, :]).argmax(-1)
      x[:, i + 1] = y
    self.backbone.reset_kv_cache()
    return x

  def _process_sigma(self, sigma):
    del sigma
    return None


class MDLM(trainer_base.AbsorbingState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()

  def _process_model_output(self, model_output, xt, sigma):
    del xt, sigma
    # zero-masking probabilities
    model_output[:, :, self.mask_index] = self.neg_infinity
    # Normalize the model_output such that x.exp() is
    # a probability distribution over vocab_size.
    # model_output = model_output.log_softmax(-1)
    return model_output

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False):
    loss_mask = xt == self.mask_index
    log_p_theta = - F.cross_entropy(
      log_x_theta[loss_mask], x0[loss_mask], reduction='none')
    loss_canvas = torch.zeros_like(x0, dtype=log_p_theta.dtype,
                                   device=self.device)
    loss_canvas[loss_mask] = log_p_theta
    if low_var:
      return - loss_canvas
    else:
      return dalpha_t / (1 - alpha_t) * loss_canvas
  
  def on_save_checkpoint(self, checkpoint):
    checkpoint['state_dict'] = collections.OrderedDict(
      (k, v) for k, v in checkpoint['state_dict'].items()
      if not k.startswith('teacher'))
    super().on_save_checkpoint(checkpoint)

  def on_load_checkpoint(self, checkpoint):
    checkpoint['state_dict'] = collections.OrderedDict(
      (k, v) for k, v in checkpoint['state_dict'].items()
      if not k.startswith('teacher'))
    super().on_load_checkpoint(checkpoint)


class EsoLM(MDLM):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self.alpha_0 = config.algo.alpha_0
    self.noise = trainer_base.LogLinear(self.alpha_0)

  def _validate_configuration(self):
    super()._validate_configuration()
    assert self.config.algo.diffusion_shuffle
    assert self.config.algo.sequential_shuffle
    assert self.config.algo.diffusion_attn_mode == 'causal'
    assert self.config.algo.sequential_attn_mode == 'causal'

  def _sort_indices(
    self, indices, shuffle, keep_masks_unshuffled=False):
    masked = (indices == self.mask_index)
    if shuffle:
      offsets = torch.rand(
        indices.shape).to(indices.device) * 0.9
      if keep_masks_unshuffled:
        # induce left-to-right order within masked tokens
        # only for sequential part
        offsets[masked] = torch.linspace(
          0, 1, torch.sum(masked)).to(indices.device)
    else:
      offsets = torch.linspace(
        0, 0.9, indices.shape[1]).to(indices.device)
    sort_idx = (masked + offsets).argsort(descending=False)
    return sort_idx

  def _loss(self, x0, valid_tokens, 
            current_accumulation_step=None, train_mode=False):
    batch_size = x0.shape[0]
    # batch size used for diffusion loss
    split_batch = int(
      self.config.algo.batch_split * batch_size)

    x0_reconstruction = x0[split_batch:]
    x0_diffusion = x0[:split_batch]
    valid_tokens_reconstruction = valid_tokens[split_batch:]
    valid_tokens_diffusion = valid_tokens[:split_batch]
    num_recons = valid_tokens_reconstruction.sum()
    num_diffusion = valid_tokens_diffusion.sum()

    do_sequential = self.config.algo.alpha_0 != 1
    do_diffusion = self.config.algo.alpha_0 != 0
    
    if do_sequential:
      assert num_recons > 0
      alpha_start = self.config.algo.alpha_0
      z0 = self.q_xt(x0_reconstruction, alpha_start)
      reconstruction_loss, sort_idx = (
        self._reconstruction_loss(x0_reconstruction, z0))
      valid_tokens_reconstruction = torch.gather(
        valid_tokens_reconstruction, dim=1, index=sort_idx)
      reconstruction_loss = (
        reconstruction_loss * valid_tokens_reconstruction).sum()
      # artificially scale the reconstruction loss so that the
      # NLL is computed correctly.
      recons_loss_per_token = reconstruction_loss / num_recons
    else:
      recons_loss_per_token = torch.tensor([0.0]).to(x0.device)

    if do_diffusion:
      assert num_diffusion > 0
      diffusion_loss, sort_idx = self.nll(
        x0_diffusion, None, current_accumulation_step, train_mode)
      valid_tokens_diffusion = torch.gather(
        valid_tokens_diffusion, dim=1, index=sort_idx)
      diffusion_loss = (
        diffusion_loss * valid_tokens_diffusion).sum()
      diffusion_loss_per_token = diffusion_loss / num_diffusion
    else:
      diffusion_loss_per_token = torch.tensor([0.0]).to(x0.device)
      
    loss_per_token = (recons_loss_per_token
                      + diffusion_loss_per_token)

    if num_recons == 0:
      num_tokens = num_diffusion
    elif num_diffusion == 0:
      num_tokens = num_recons
    else:
      num_tokens = num_diffusion
    
    return trainer_base.Loss(
        loss=loss_per_token,
        nlls=loss_per_token * num_tokens,
        reconstruction_loss=recons_loss_per_token * num_tokens,
        num_tokens=num_tokens)

  def _reconstruction_loss(self, x0, z0):
    dummy_t0 = torch.zeros(1, z0.shape[0], dtype=self.dtype,
                           device=self.device)
    # sort inputs and targets before passing to the model
    sort_idx = self._sort_indices(
      z0, shuffle=self.config.algo.sequential_shuffle,
      keep_masks_unshuffled=True)
    z0 = torch.gather(z0, dim=1, index=sort_idx)
    x0 = torch.gather(x0, dim=1, index=sort_idx)
    # pass sort_idx into the model to also sort pos. embeddings   
    # _process_model_output performs zero-masking trick 
    model_output_t0 = self.forward(
      z0, dummy_t0, sort_idx, x0=x0)
    reconstruction_loss = - torch.gather(
      input=model_output_t0,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    # carry-over loss masking
    loss_mask = z0 == self.mask_index
    reconstruction_loss = reconstruction_loss * loss_mask
    return reconstruction_loss, sort_idx

  def nll(self, x0, output_tokens,
          current_accumulation_step=None, train_mode=False):
    del output_tokens
    t = self._sample_t(x0.shape[0],
                       current_accumulation_step)
    assert t.shape[0] == x0.shape[0]
    if self.T > 0:
      t = (t * self.T).to(torch.int)
      t = t / self.T
      # t \in {1/T, 2/T, ..., 1}
      t += (1 / self.T)
    
    dalpha_t, alpha_t = self.noise(t)
    alpha_t = alpha_t.unsqueeze(-1)
    assert alpha_t.ndim == 2
    sigma = self._sigma_from_alphat(alpha_t)

    xt = self.q_xt(x0, alpha_t)
    # sort inputs and targets before passing to the model
    sort_idx = self._sort_indices(
      xt, shuffle=self.config.algo.diffusion_shuffle)
    xt = torch.gather(xt, dim=1, index=sort_idx)
    x0 = torch.gather(x0, dim=1, index=sort_idx)
    # pass sort_idx into the model to also sort pos. embeddings
    # _process_model_output performs zero-masking trick
    log_x_theta = self.forward(xt, sigma=sigma, sort_idx=sort_idx)
    # nll_per_token performs carry-over loss masking
    return self.nll_per_token(
      log_x_theta=log_x_theta,
      xt=xt,
      x0=x0,
      alpha_t=alpha_t,
      dalpha_t=dalpha_t,
      low_var=train_mode and self.loss_type == 'low_var'), sort_idx
  
  def _sample_t(self, n, accum_step):
    if accum_step is not None:
      # During training
      batch_dim = n
      n = int(self.config.loader.global_batch_size
              * self.config.algo.batch_split)
    _eps_t = torch.rand(n, device=self.device)
    if self.antithetic_sampling:
      offset = torch.arange(n, device=self.device) / n
      _eps_t = (_eps_t / n + offset) % 1
    t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
    if accum_step is not None:
      t = t.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
      t = t.chunk(self.trainer.num_devices)[self.trainer.local_rank]
      t = t.chunk(self.trainer.accumulate_grad_batches)[
        accum_step]
      # corner case for the last datapoint
      t = t[:batch_dim]
    return t

  def _tokens_unmasked_per_step(self, num_steps):
    remaining_tokens = self.num_tokens
    num_tokens_to_unmask = []
    dt = 1 / num_steps
    # Assumes a log-linear schedule.
    for t in np.linspace(1, dt, num_steps):
      _, alpha_t = self.noise(t)
      _, alpha_s = self.noise(t - dt)
      n_unmask = np.random.binomial(
        remaining_tokens, (alpha_s - alpha_t) / (1 - alpha_t))
      if n_unmask != 0:
        num_tokens_to_unmask.append(n_unmask)
        remaining_tokens -= n_unmask
    if remaining_tokens != 0 and self.alpha_0 == 1:
      num_tokens_to_unmask.append(remaining_tokens)
    return num_tokens_to_unmask

  def _get_sampler_config(self, num_steps, num_samples):
    if self.config.sampling.predictor == 'ancestral':
      unmask_k_tokens = self._tokens_unmasked_per_step(num_steps)
      num_diffusion_tokens = sum(unmask_k_tokens)
      
      sort_idx = torch.rand(
        num_samples, self.num_tokens).argsort(
          descending=False).to(self.device)
      # Diffusion Tokens: shuffle
      # Sequential Tokens: don't shuffle
      sort_idx[:, num_diffusion_tokens:] = (
        sort_idx[:, num_diffusion_tokens:].sort().values)
      unmask_k_tokens = unmask_k_tokens + [1] * (
        self.num_tokens - num_diffusion_tokens)
    elif self.config.sampling.predictor == 'block_autoregressive':
      unmask_k_tokens, sort_idx = _block_autoregressive_config(
        sub_context_length=num_steps,
        sar_steps=self.config.sampling.sar_steps,
        num_parallel_decode=self.num_tokens // (
          self.config.sampling.sar_steps * num_steps),
        num_samples=num_samples,
        device=self.device)
      assert sum(unmask_k_tokens) == self.num_tokens
    elif self.config.sampling.predictor == 'block':
      unmask_k_tokens, sort_idx = _block_sampler_config(
        sub_context_length=num_steps,
        num_parallel_decode=self.num_tokens // num_steps,
        num_samples=num_samples,
        device=self.device)
      assert sum(unmask_k_tokens) == self.num_tokens
    return unmask_k_tokens, sort_idx

  @torch.no_grad()
  def generate_samples(self, num_samples, num_steps=None,
                       eps=1e-5):
    """
    Generate samples from the model (only supports Eso-LM (B)).
    """
    if num_steps is None:
      num_steps = self.config.sampling.steps

    (unmask_k_tokens, sort_idx) = self._get_sampler_config(
      num_steps=num_steps, num_samples=num_samples)
  
    x = self.prior_sample(num_samples, self.num_tokens)

    assert sum(unmask_k_tokens) == self.num_tokens
    noise = torch.distributions.Gumbel(0, 1).sample(
      (num_samples, self.num_tokens,
       self.vocab_size)).to(self.device)
    unmasked_tokens = 0
    kv_cache = self.config.sampling.kv_cache
    self.backbone.reset_kv_cache()
    last_k_start = 0
    for i, k in enumerate(unmask_k_tokens):
      if i > 0:
        last_k_start = unmasked_tokens - unmask_k_tokens[i - 1]
      with torch.amp.autocast('cuda', dtype=torch.float32):
        log_p_x0 = self.backbone.forward_sample(
          zt=x,
          sort_idx=sort_idx,
          kv_cache=kv_cache,
          last_k_start=last_k_start,
          curr_k_start=unmasked_tokens,
          curr_k_end=unmasked_tokens + k)
      log_p_x0[:, :, self.mask_index] = self.neg_infinity
      if self.config.sampling.p_nucleus < 1:
        log_p_x0 = utils.top_k_top_p_filtering(
          log_p_x0, top_p=self.config.sampling.p_nucleus)
      indices = slice(unmasked_tokens, unmasked_tokens + k)
      if kv_cache:
        y = (log_p_x0 + noise[:, indices, :]).argmax(-1)
      else:
        y = (log_p_x0[:, indices, :]
             + noise[:, indices, :]).argmax(-1)
      x[:, indices] = y
      unmasked_tokens += k
    self.backbone.reset_kv_cache()
    sort_idx_reversed = utils.get_reverse_indices(sort_idx)
    x = torch.gather(x, dim=1, index=sort_idx_reversed)
    return x


class DUO_BASE(trainer_base.UniformState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()

  def on_save_checkpoint(self, checkpoint):
    checkpoint['state_dict'] = collections.OrderedDict(
      (k, v) for k, v in checkpoint['state_dict'].items()
      if not k.startswith('teacher'))
    super().on_save_checkpoint(checkpoint)

  def on_load_checkpoint(self, checkpoint):
    checkpoint['state_dict'] = collections.OrderedDict(
      (k, v) for k, v in checkpoint['state_dict'].items()
      if not k.startswith('teacher'))
    super().on_load_checkpoint(checkpoint)

  def _process_model_output(self, model_output, xt, sigma):
    del xt, sigma
    model_output[:, :, self.mask_index] = self.neg_infinity
    return model_output.log_softmax(dim=-1)

  def _compute_posterior(self, x, xt, alpha_s, alpha_t):
    """Computes the posterior / approximate posterior.

    Args:
      x: Either clean input `x0` (one-hot),
        or model's predicted `x_theta` of shape (B, L, V).
      xt: The noisy latent (as indices) of shape (B, L).
      alpha_s: Noise level at s of shape (B, [L | 1], 1).
      alpha_t: Noise level at t of shape (B, [L | 1], 1).

    Returns:
      Posterior / approximate posterior of shape (B, L, V).
    """
    if self.config.sampling.use_float64:
      x = x.to(torch.float64)
    if alpha_s.ndim == 2:
      alpha_s = alpha_s.unsqueeze(-1)
    if alpha_t.ndim == 2:
      alpha_t = alpha_t.unsqueeze(-1)
    alpha_ts = alpha_t / alpha_s
    d_alpha = alpha_s - alpha_t
    xt_one_hot = F.one_hot(xt, self.vocab_size).to(
      self.dtype).to(self.device)
    return (
      (alpha_t * self.vocab_size * x * xt_one_hot + (
        alpha_ts - alpha_t) * xt_one_hot + d_alpha * x + (
          1 - alpha_ts) * (1 - alpha_s) / self.vocab_size) / (
            alpha_t * self.vocab_size * torch.gather(
              x, -1, xt[..., None]) + (1 - alpha_t)))

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False):
    if log_x_theta.shape[1] == self.num_tokens:
      return self._nll_per_token_torch_compile(
        log_x_theta, xt, x0, alpha_t, dalpha_t, low_var)
    return self._nll_per_token(log_x_theta, xt, x0, alpha_t,
                               dalpha_t, low_var)

  @torch.compile
  def _nll_per_token_torch_compile(self, *args, **kwargs):
    return self._nll_per_token(*args, **kwargs)

  def _nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                     dalpha_t, low_var=False):
    del low_var
    assert alpha_t.ndim == 2
    assert x0.ndim == 2
    assert xt.ndim == 2
    assert not torch.is_tensor(dalpha_t) or dalpha_t.ndim == 2
    x_reconst = log_x_theta.exp()
    x_bar_theta = self.vocab_size * alpha_t[
        :, :, None] * x_reconst + 1 - alpha_t[:, :, None]
    coeff = dalpha_t / (self.vocab_size * alpha_t)
    x_eq_xt = (x0 == xt).float()
    x_neq_xt = 1 - x_eq_xt
    xbar_xt = (1 - alpha_t) + self.vocab_size * alpha_t * x_eq_xt
    xbar_theta_xt = torch.gather(
      x_bar_theta, -1, xt.unsqueeze(-1)).squeeze(-1)
    xbar_theta_x = torch.gather(
      x_bar_theta, -1, x0.unsqueeze(-1)).squeeze(-1)
    term1 = self.vocab_size * (1 / xbar_xt
                                - 1 / xbar_theta_xt)
    
    const = (1 - alpha_t) / (self.vocab_size * alpha_t
                             + 1 - alpha_t)
    term2_coefs = x_eq_xt * const + x_neq_xt
    term2_offset = ((self.vocab_size - 1) * const * x_eq_xt
                    - (1 / const) * x_neq_xt) * const.log()
    term2_theta = - term2_coefs * (
      x_bar_theta.log().sum(-1)
      - self.vocab_size * xbar_theta_xt.log())
    term2_theta = (
      term2_theta
      - self.vocab_size * alpha_t / (1 - alpha_t) * (
        xbar_theta_x.log() - xbar_theta_xt.log()) * x_neq_xt)
    term2 = term2_theta + term2_offset
    diffusion_loss = coeff * (term1 - term2)
    assert diffusion_loss.ndim == 2
    return diffusion_loss

  def _ancestral_update(self, x, t, dt, p_x0=None,
                   noise_removal_step=False):
    del p_x0
    _, alpha_t = self.noise(t)
    if noise_removal_step:
      alpha_s = torch.ones_like(alpha_t)
    else:
      _, alpha_s = self.noise(t - dt)
    sigma_t = self._sigma_from_alphat(alpha_t)
    assert alpha_t.ndim == 2
    
    q_xs = self._compute_posterior(
      x=self.forward(x, sigma_t).exp(),
      xt=x,
      alpha_s=alpha_s,
      alpha_t=alpha_t)
    if self.config.sampling.use_float64:
      q_xs = q_xs.to(torch.float64)
    if self.p_nucleus < 1:
      log_p_x0 = utils.top_k_top_p_filtering(
        q_xs.log(), top_p=self.p_nucleus)
      noise = torch.distributions.Gumbel(0, 1).sample(
        log_p_x0.shape).to(self.device)
      return None, (log_p_x0 + noise).argmax(dim=-1)
    return None, trainer_base.sample_categorical(q_xs)
