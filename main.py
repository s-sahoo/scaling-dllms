import itertools
import json
import os
import time

import calflops
import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch

import algo
import dataloader
import utils

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)
omegaconf.OmegaConf.register_new_resolver(
  'max', lambda x, y: max(x, y))
omegaconf.OmegaConf.register_new_resolver(
  'utils.flops_to_max_steps', utils.flops_to_max_steps)


def _compute_flops(diffusion_model, config, tokenizer,
                   logger):
  logger.info('Starting FLOPS Computation.')
  # N.B: Use flex_attention_compiled = flex_attention in models/dit.py
  flops, macs, params = calflops.calculate_flops(
    model=diffusion_model(config, tokenizer=tokenizer),
    args=[],
    kwargs={'xt': torch.ones(1, config.model.length,
                              dtype=torch.int32,
                              device='cuda'),
            'sigma': torch.zeros(1, 1, device='cuda')},
    output_as_string=True,
    output_precision=4)
  flops, scale = flops.split(' ')
  if scale == 'GFLOPS':
    flops = float(flops) * 1e9
  elif scale == 'TFLOPS':
    flops = float(flops) * 1e12
  else:
    raise ValueError(f'Invalid scale: {scale}')
  
  if params.endswith('M'):
    params = int(float(params.replace('M', '')) * 1e6)
  elif params.endswith('B'):
    params = int(float(params.replace('B', '')) * 1e9)
  else:
    raise ValueError(f'Invalid scale: {params}')
  
  logger.info(f'Vocab size: {len(tokenizer)}')
  logger.info(f'FLOPS: {flops}')
  logger.info(f'MACS: {macs}')
  logger.info(f'PARAMS: {params}')
  non_embed_params = (
    params - 2 * len(tokenizer) * config.model.hidden_size)
  logger.info(f'Non-embed params: {non_embed_params}')
  
  samples_path = config.eval.generated_samples_path
  with fsspec.open(samples_path, 'w') as f:
    data = {
      'vocab_size': len(tokenizer),
      'non_embed_params': non_embed_params,
      'flops': flops,
      'macs': macs,
      'params': params,
      'context_length': config.model.length,
      'model_hidden_size': config.model.hidden_size,
      'model_name': config.model.name,
      'model_type': config.model.type,
      'model_n_blocks': config.model.n_blocks,
      'model_n_heads': config.model.n_heads,
    }
    json.dump(data, f, ensure_ascii=False, indent=4)
    print('Samples saved at:', samples_path)


def _load_from_checkpoint(diffusion_model, config, tokenizer):
  if 'hf' in config.algo.backbone:
    return diffusion_model(
      config, tokenizer=tokenizer).to('cuda')
  
  return diffusion_model.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config,
    map_location='cpu').to('cuda')


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    if type(batch) == dict:
      batch = batch['input_ids']
    print('Batch input_ids.shape', batch.shape)
    first = batch[0, :k]
    last = batch[0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)


def _instantiate_trainer(diffusion_model, config, logger, tokenizer):
  model = _load_from_checkpoint(
    diffusion_model=diffusion_model,
    config=config,
    tokenizer=tokenizer)

  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  return trainer, model


def _generate_samples(diffusion_model, config, logger,
                      tokenizer):
  logger.info('Starting Sample Eval.')
  trainer, model = _instantiate_trainer(
    diffusion_model=diffusion_model,
    config=config,
    logger=logger,
    tokenizer=tokenizer)
  start_time = time.time()
  samples = trainer.predict(model)
  generative_ppl = model.metrics.gen_ppl.compute().item()
  entropy = model.metrics.sample_entropy.compute().item()
  time_elapsed = (time.time() - start_time) / 60
  if trainer.global_rank == 0:
    print('Generative perplexity:', generative_ppl)
    print('Sample entropy:', entropy)
    print(f'Time taken: {time_elapsed:.1f} minutes')
  samples_path = config.eval.generated_samples_path.replace(
    '.json', f'_gpu-{trainer.global_rank}.json')
  with fsspec.open(samples_path, 'w', encoding='utf-8') as f:
    alpha_0 = -1
    if config.algo.name == 'esolm':
      alpha_0 = config.algo.alpha_0
    data = {
      'T': config.sampling.steps,
      'model': config.model.name,
      'sar_steps': config.sampling.sar_steps,
      'p_nucleus': config.sampling.p_nucleus,
      'alpha_0': alpha_0,
      'predictor': config.sampling.predictor,
      'noise_removal': config.sampling.noise_removal,
      'generative_ppl': generative_ppl,
      'entropy': entropy,
      'time_elapsed_in_mins': time_elapsed,
      # Flatten the list of lists
      'generated_seqs': list(itertools.chain.from_iterable(samples))}
    json.dump(data, f, ensure_ascii=False, indent=4)
  print('Samples saved at:', samples_path)


def _eval_ppl(diffusion_model, config, logger, tokenizer):
  logger.info('Starting Perplexity Eval.')
  trainer, model = _instantiate_trainer(
    diffusion_model=diffusion_model,
    config=config,
    logger=logger,
    tokenizer=tokenizer)
  _, valid_ds = dataloader.create_dataloaders(
    config,
    seed=config.seed + trainer.global_rank,
    current_device=trainer.global_rank,
    total_devices=trainer.num_nodes * trainer.num_devices)
  trainer.validate(model, valid_ds)


def _train(diffusion_model, config, logger, tokenizer):
  logger.info('Starting Training.')
  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      **config.wandb)

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  ckpt_path = None
  load_ckpt_path = None
  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
    load_ckpt_path = ckpt_path

  if config.training.finetune_path != '':
    assert utils.fsspec_exists(config.training.finetune_path)
    load_ckpt_path = config.training.finetune_path

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)

  if load_ckpt_path is None:
    model = diffusion_model(config, tokenizer=tokenizer)
  else:
    # Load checkpoint to CPU first to avoid GPU memory issues
    # in distributed training
    model = diffusion_model.load_from_checkpoint(
      load_ckpt_path,
      tokenizer=tokenizer,
      config=config,
      map_location='cpu')
  
  train_ds, valid_ds = dataloader.create_dataloaders(
    config,
    seed=config.seed + trainer.global_rank,
    train_start_file_idx=model.train_start_file_idx,
    current_device=trainer.global_rank,
    total_devices=trainer.num_nodes * trainer.num_devices)
  
  # Log node and rank information
  logger.info('Current node: {}, Total nodes: {}'.format(
    trainer.node_rank, trainer.num_nodes))
  logger.info('Local rank: {}, Global rank: {}'.format(
    trainer.local_rank, trainer.global_rank))
  
  _print_batch(train_ds, valid_ds, tokenizer)
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)
  trainer.validate(model, valid_ds)

@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)
  if config.algo.name == 'ar':
    diffusion_model = algo.AR
  elif config.algo.name == 'mdlm':
    diffusion_model = algo.MDLM
  elif config.algo.name == 'esolm':
    diffusion_model = algo.EsoLM
  elif config.algo.name == 'duo_base':
    diffusion_model = algo.DUO_BASE
  else:
    raise ValueError(
      f'Invalid algorithm name: {config.algo.name}')
  kwargs = {'diffusion_model': diffusion_model,
            'config': config,
            'tokenizer': tokenizer,
            'logger': logger}
  if config.mode == 'sample_eval':
    _generate_samples(**kwargs)
  elif config.mode == 'ppl_eval':
    _eval_ppl(**kwargs)
  elif config.mode == 'compute_flops':
    _compute_flops(**kwargs)
  else:
    _train(**kwargs)


if __name__ == '__main__':
  main()