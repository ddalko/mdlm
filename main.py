import json
import os

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch

import dataloader
import diffusion
import utils
from json_utils import CodeBlockJsonParser, extract_pred, validate

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def load_model(config):
  tokenizer = dataloader.get_tokenizer(config)
  
  # Lightning 체크포인트에서 로드하는 경우 (.ckpt 파일)
  if hasattr(config.eval, 'checkpoint_path') and config.eval.checkpoint_path.endswith('.ckpt'):
    return diffusion.Diffusion.load_from_checkpoint(
      config.eval.checkpoint_path,
      tokenizer=tokenizer,
      config=config)
  
  # HuggingFace 모델 직접 로드하는 경우
  elif config.backbone == 'hf_dit':
    return diffusion.Diffusion(config, tokenizer=tokenizer).to('cuda')
  
  # 기본 동작 (새로 모델 생성)
  else:
    return diffusion.Diffusion(config, tokenizer=tokenizer).to('cuda')


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
def _print_batch(train_ds, valid_ds, tokenizer, k=512):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)


def generate_samples(config, logger, tokenizer):
  logger.info('Generating samples.')
  model = load_model(config)
  model.gen_ppl_metric.reset()
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  stride_length = config.sampling.stride_length
  num_strides = config.sampling.num_strides
  for _ in range(config.sampling.num_sample_batches):
    if config.sampling.semi_ar:
      _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
        stride_length=stride_length,
        num_strides=num_strides,
        dt=1 / config.sampling.steps)
      text_samples = intermediate_samples[-1]
      # Note: Samples generated using semi-ar method
      # need to to be processed before computing generative perplexity
      # since these samples contain numerous <|endoftext|> tokens
      # and diffusion.compute_generative_perplexity() discards
      # any text after the first EOS token.
    else:
      samples = model.restore_model_and_sample(
        num_steps=config.sampling.steps)
      text_samples = model.tokenizer.batch_decode(samples)
      model.compute_generative_perplexity(text_samples)
  print('Text samples:', text_samples)
  if not config.sampling.semi_ar:
    print('Generative perplexity:',
          model.gen_ppl_metric.compute())
  return text_samples

def _ppl_eval(config, logger, tokenizer):
  logger.info('Starting Zero Shot Eval.')

  model = load_model(config)
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
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=config.seed)
  trainer.validate(model, valid_ds)


def _json_eval(config, logger, tokenizer):
  logger.info('Eval JSON samples.')
  model = load_model(config)
  train_ds, valid_ds = dataloader.get_dataloaders(
    config, tokenizer) 
  _print_batch(train_ds, valid_ds, tokenizer)
  codeblockjsonparser = CodeBlockJsonParser()

  score = 0
  total_samples = 0
  results = []
  for batch_idx, batch in enumerate(valid_ds):
    model.backbone.eval()
    model.noise.eval()

    prompt_tokens = batch['input_ids'].to('cuda')
    prompt_masks = batch['prompt_mask'].to(device='cuda', dtype=torch.bool)

    # 마지막 True 하나만 False 로 변경 (배치 처리 지원)
    if prompt_masks.dim() == 2:
      # 마지막 True 위치만 True 인 마스크 생성
      # cumsum == total_sum 이 되는 최초 지점이 마지막 True
      last_true_mask = prompt_masks & (prompt_masks.cumsum(-1) == prompt_masks.sum(-1, keepdim=True))
      prompt_masks[last_true_mask] = False
    else:
      # 1D 인 경우
      true_indices = torch.nonzero(prompt_masks, as_tuple=False).squeeze(-1)
      if true_indices.numel() > 0:
        prompt_masks[true_indices[-1]] = False

    if getattr(config, 'json_structure_token_prompting', False):
      logger.info('Using JSON structure token prompting.')
      json_structure_masks = batch['json_structure_mask'].to(device=prompt_masks.device, dtype=torch.bool)
    else:
      json_structure_masks = torch.zeros_like(prompt_masks, device=prompt_masks.device, dtype=torch.bool)

    samples = model.restore_model_and_sample_with_prompt(
      prompt_tokens=prompt_tokens,
      prompt_masks=prompt_masks,
      json_structure_masks=json_structure_masks,
      num_steps=config.sampling.steps,
      eps=config.training.sampling_eps,
    )

    # Remove token ID 102 before decoding
    filtered_samples = []
    for sample in samples:
      # Filter out token ID 102 from each sequence
      filtered_sample = sample[sample != 102]
      filtered_samples.append(filtered_sample)
    
    # Convert to list for individual decoding (avoids padding issues)
    text_samples = []
    for filtered_sample in filtered_samples:
      decoded_text = model.tokenizer.decode(filtered_sample, skip_special_tokens=True)
      text_samples.append(decoded_text)
    
    # Optional: Additional string-level cleanup if needed
    # If token 102 has a specific string representation, uncomment below:
    # token_102_str = tokenizer.decode([102])
    # text_samples = [text.replace(token_102_str, '') for text in text_samples]
    for idx, seq in enumerate(text_samples):
      tmp = {"id": total_samples, "error_msg": ""}
      total_samples += 1
      correct = False
      pred_json = None
      try:
        prompt = batch['prompt'][idx]
        json_schema = codeblockjsonparser.loads(prompt) if not config.ignore_validate_schema else {}
        pred_json = extract_pred(seq)
        if validate(pred_json, verify_schema=json_schema):
          print(f"{idx}: ✅ JSON valid & correct!")
          score += 1
          correct = True
      except Exception as e:
        print(f"{idx}: ❌ JSON invalid or incorrect. Error: {e}")
        if pred_json is not None:
          pred = pred_json
        tmp["error_msg"] = str(e)
        response = torch.masked_select(batch['input_ids'][idx].to(prompt_masks.device), ~prompt_masks[idx])
        response = response[response != 102]
        response = response[response != 50256]
        response = response[response != 50257]
        tmp["gt"] = tokenizer.decode(response)
      
      tmp.update({
        "pred": pred,
        "schema": json_schema,
        "correct": correct,
      })
      results.append(tmp)

    print(f'JSON Validity: {score / total_samples:%}')
    with open("json_eval_results.json", "w") as f:
      json.dump(results, f, indent=2)


def _train(config, logger, tokenizer):
  logger.info('Starting Training.')
  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  train_ds, valid_ds = dataloader.get_dataloaders(
    config, tokenizer)
  _print_batch(train_ds, valid_ds, tokenizer)

  model = diffusion.Diffusion(
    config, tokenizer=valid_ds.tokenizer)

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)

  if config.mode == 'sample_eval':
    generate_samples(config, logger, tokenizer)
  elif config.mode == 'ppl_eval':
    _ppl_eval(config, logger, tokenizer)
  elif config.mode == 'json_eval':
    _json_eval(config, logger, tokenizer)
  else:
    _train(config, logger, tokenizer)


if __name__ == '__main__':
  main()