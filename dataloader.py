import functools
import itertools
import json
import math
import os
import re
import shutil
import typing
import urllib
import zipfile

import datasets
import fsspec
import requests
import tokenizers
import torch
import transformers

import utils

LOGGER = utils.get_logger(__name__)

# for GPT2 tokenizer
JSON_STRUCTURE_TOKEN_IDS = [1, 11, 25, 58, 60, 90, 92, 366, 553, 1298, 1600, 2404, 2430, 3712, 4357, 4895, 5512, 5974, 7131, 8351, 8762, 8973, 9063, 9832, 11097, 11709, 11907, 11919, 13018, 14692, 15931, 17241, 17414, 17912, 18477, 20598, 20662, 21737, 23846, 24022, 25719, 26358, 27007, 29164, 30109, 30866, 32509, 33116, 33250, 34171, 34713, 36786, 37811, 38362, 38430, 42535, 42785, 43661, 45299, 47182, 47682, 47715, 48999]

def get_json_structure_mask(tokens):
  """
  Use predefined JSON_STRUCTURE_TOKEN_IDS from dataloader.py to identify structure tokens.
  This is the most reliable approach using the ground truth token IDs.
  
  Args:
      text: Full text containing prompt and response

  Returns:
      tuple: structure_mask
  """
  # Create structure mask - only mark tokens that are in JSON_STRUCTURE_TOKEN_IDS
  structure_mask = torch.zeros(len(tokens), dtype=torch.bool)
  structure_token_set = set(JSON_STRUCTURE_TOKEN_IDS)
  
  for i, token in enumerate(tokens):
    # token_id = token.item()
    token_id = token
    if token_id in structure_token_set:
      # This is a structure token
      structure_mask[i] = True

  return structure_mask

def wt_detokenizer(string):
  # contractions
  string = string.replace("s '", "s'")
  string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
  # number separators
  string = string.replace(" @-@ ", "-")
  string = string.replace(" @,@ ", ",")
  string = string.replace(" @.@ ", ".")
  # punctuation
  string = string.replace(" : ", ": ")
  string = string.replace(" ; ", "; ")
  string = string.replace(" . ", ". ")
  string = string.replace(" ! ", "! ")
  string = string.replace(" ? ", "? ")
  string = string.replace(" , ", ", ")
  # double brackets
  string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
  string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
  string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
  string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
  string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
  # miscellaneous
  string = string.replace("= = = =", "====")
  string = string.replace("= = =", "===")
  string = string.replace("= =", "==")
  string = string.replace(" " + chr(176) + " ", chr(176))
  string = string.replace(" \n", "\n")
  string = string.replace("\n ", "\n")
  string = string.replace(" N ", " 1 ")
  string = string.replace(" 's", "'s")
  return string


def ptb_detokenizer(x):
  x = x.replace(" 's", "'s")
  x = x.replace("s ' ", "s' ")
  x = x.replace(" n't", "n't")
  x = x.replace(" \n ", "\n")
  x = x.replace("\\/", "/")
  for _ in range(10):
      x = x.replace(" N ", " 1 ")
  x = x.replace("$ 1", "$1")
  x = x.replace("# 1", "#1")
  x = x.replace("<unk>", "?")
  return x


def lm1b_detokenizer(x):
  x = x.replace('http : / / ', 'http://')
  x = x.replace('https : / / ', 'https://')
  x = re.sub(r' \'(\w+)', r"'\1", x)
  x = re.sub(r' (\w+) \. ', r' \1. ', x)
  x = re.sub(r' (\w+) \.$', r' \1.', x)
  x = x.replace(' ? ', '? ')
  x = re.sub(r' \?$', '?', x)
  x = x.replace(' ! ', '! ')
  x = re.sub(r' \!$', '!', x)
  x = x.replace(' , ', ', ')
  x = x.replace(' : ', ': ')
  x = x.replace(' ; ', '; ')
  x = x.replace(' / ', '/')
  x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
  x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
  x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
  x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
  x = x.replace('$ ', '$')
  x = x.replace('£ ', '£')
  return x


def lambada_detokenizer(text):
  text = text.replace("“", '"')
  text = text.replace("”", '"')
  return '\n'+text.strip()


def scientific_papers_detokenizer(x):
  x = wt_detokenizer(x)
  x = lm1b_detokenizer(x)
  return x


class Text8Tokenizer(transformers.PreTrainedTokenizer):
  def __init__(
    self,
    bos_token='[BOS]',
    eos_token='[EOS]',
    sep_token='[SEP]',
    cls_token='[CLS]',
    pad_token='[PAD]',
    mask_token='[MASK]',
    unk_token='[UNK]',
    **kwargs):
    self.characters = list('abcdefghijklmnopqrstuvwxyz ')
    self._vocab_str_to_int = {
      '[CLS]': 0,
      '[SEP]': 1,
      '[BOS]': 2,
      '[EOS]': 3,
      '[MASK]': 4,
      '[PAD]': 5,
      '[RESERVED]': 6,
      '[UNK]': 7,
      ** {ch: i + 8 for i, ch in enumerate(self.characters)}}
    self._vocab_int_to_str = {
      v: k for k, v in self._vocab_str_to_int.items()}
    super().__init__(
      bos_token=bos_token,
      eos_token=eos_token,
      sep_token=sep_token,
      cls_token=cls_token,
      pad_token=pad_token,
      mask_token=mask_token,
      unk_token=unk_token,
      **kwargs)

  @property
  def vocab_size(self) -> int:
    return len(self._vocab_str_to_int)

  def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
    return list(text.lower())

  def _convert_token_to_id(self, token: str) -> int:
    return self._vocab_str_to_int.get(
      token, self._vocab_str_to_int['[UNK]'])

  def _convert_id_to_token(self, index: int) -> str:
    return self._vocab_int_to_str[index]

  def convert_tokens_to_string(self, tokens):
    return ''.join(tokens)

  def get_vocab(self) -> typing.Dict[str, int]:
    return self._vocab_str_to_int

def generate_masks_token_based(example, tokenizer, block_size):
  """
  Generate both prompt and JSON structure masks using token-based approach.
  
  Args:
    example: Dataset example containing 'text' and 'prompt' fields
    tokenizer: Tokenizer to use for processing
    block_size: Maximum sequence length
    
  Returns:
    tuple: (prompt_masks, json_structure_masks)
  """
  # Handle batched processing
  texts = example['text'] if isinstance(example['text'], list) else [example['text']]
  prompts = example['prompt'] if isinstance(example['prompt'], list) else [example['prompt']]
  
  if len(texts) != len(prompts):
    # If single prompt for multiple texts, repeat it
    if len(prompts) == 1:
      prompts = prompts * len(texts)
    else:
      raise ValueError(f"Mismatch between texts ({len(texts)}) and prompts ({len(prompts)})")
  
  prompt_masks = []
  json_structure_masks = []
  
  for text, prompt in zip(texts, prompts):
    # Tokenize full text and prompt
    text_tokens = tokenizer(text, 
                           max_length=block_size,
                           padding='max_length', 
                           truncation=True,
                           add_special_tokens=True)['input_ids']
    
    prompt_tokens = tokenizer(prompt,
                             add_special_tokens=True)['input_ids']
    
    # Find where prompt ends in text_tokens
    prompt_mask = [False] * len(text_tokens)
    json_structure_mask = [False] * len(text_tokens)
    
    # Simple approach: mark the first len(prompt_tokens) as prompt
    # This assumes prompt is at the beginning
    prompt_len = min(len(prompt_tokens), len(text_tokens))
    for i in range(prompt_len):
      prompt_mask[i] = True
    
    # Get response tokens (everything after prompt)
    response_start_idx = prompt_len
    response_tokens = text_tokens[response_start_idx:]
    
    # Filter out padding tokens from response
    response_tokens = [t for t in response_tokens if t != tokenizer.pad_token_id]
    
    # Get JSON structure mask for response tokens
    if response_tokens:
      response_structure_mask = get_json_structure_mask(response_tokens)
      
      # Map back to full text_tokens
      for i, is_structure in enumerate(response_structure_mask):
        full_idx = response_start_idx + i
        if full_idx < len(json_structure_mask):
          json_structure_mask[full_idx] = is_structure
    
    prompt_masks.append(prompt_mask)
    json_structure_masks.append(json_structure_mask)
  
  return prompt_masks, json_structure_masks


def process_schemabench_masks(example, tokens, tokenizer, block_size):
  """
  Process and add masks for schemabench dataset using token-based approach.
  
  Args:
    example: Dataset example containing 'text' and 'prompt' fields
    tokens: Existing tokens dictionary to modify
    tokenizer: Tokenizer to use for processing
    block_size: Maximum sequence length
  """
  try:
    prompt_masks, json_structure_masks = generate_masks_token_based(example, tokenizer, block_size)
    tokens['prompt_mask'] = prompt_masks
    tokens['json_structure_mask'] = json_structure_masks
  except Exception as e:
    LOGGER.warning(f"Failed to generate masks for schemabench: {e}")
    # Fallback: create empty masks
    batch_size = len(example['text']) if isinstance(example['text'], list) else 1
    tokens['prompt_mask'] = [[False] * block_size for _ in range(batch_size)]
    tokens['json_structure_mask'] = [[False] * block_size for _ in range(batch_size)]

def get_lambada_test_dataset():
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url):
      response = requests.get(url, stream=True)
      data_list = []

      # Process each line in the response content
      for line in response.iter_lines(decode_unicode=True):
        if line:
          data = json.loads(line)
          data_list.append(data)

      return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = datasets.Dataset.from_list(lambada_data)
    return dataset

def get_text8_dataset(cache_dir, max_seq_length=256,
                      drop_last=True, crop_train=False):
  """Adapted from:
    https://github.com/google-research/google-research/blob/master/d3pm/text/datasets.py#L344

    Args:
      cache_dir: str, path to cache directory.
      max_seq_length: int, maximum length of sequences.
          (default: 256, as in D3PM codebase.)
      drop_last: bool, whether to drop the last incomplete
          batch. (default: True, as in D3PM codebase.)
      crop_train: bool, whether to subsample contiguous
          subsequences from training example. serves to
          make sure transformer models with absolute position
          embeddings do not have incorrect position-wise
          marginals. (default: False, but necessary to match D3PM AR)

    Returns:
      dataset: dataset.DatasetDict, with keys 'train',
          'valid', 'test'.
  """
  url = 'http://mattmahoney.net/dc/text8.zip'
  if not crop_train:
    cache_dir = f'{cache_dir}/text8'
  else:
    cache_dir = f'{cache_dir}/text8-crop-train'
  split_names = ['train', 'validation', 'test']
  if not all([
    utils.fsspec_exists(os.path.join(cache_dir, split))
    for split in split_names
  ]):
    # Check if raw data exists
    raw_cache_dir = os.path.join(cache_dir, 'raw_data')
    if not all([
      utils.fsspec_exists(
        os.path.join(raw_cache_dir, f'text8.{split}.txt'))
      for split in split_names
    ]):
      if not utils.fsspec_exists(
        os.path.join(raw_cache_dir, 'text8.zip')):
        utils.fsspec_mkdirs(raw_cache_dir, exist_ok=True)
        LOGGER.info('Downloading text8 from URL {}.'.format(url))
        with (urllib.request.urlopen(url) as in_stream,
              open(os.path.join(raw_cache_dir, 'text8.zip'),
                   'wb') as out_file):
          shutil.copyfileobj(in_stream, out_file)

      with fsspec.open(
        os.path.join(raw_cache_dir, 'text8.zip'),
        'rb') as f:
        rawdata = zipfile.ZipFile(f).read(
          'text8').decode('utf-8')

      # Splits taken from D3PM codebase
      splits = {
        'train': rawdata[:90000000],
        'validation': rawdata[90000000: 95000000],
        'test': rawdata[95000000:],
      }

      for split, data in splits.items():
        _path = os.path.join(raw_cache_dir,
                             f'text8.{split}.txt')
        with fsspec.open(_path, 'w') as f:
          f.write(data)
    else:
      splits = {}
      for split in split_names:
        _path = os.path.join(raw_cache_dir,
                             f'text8.{split}.txt')
        with fsspec.open(_path, 'r') as f:
          splits[split] = f.read()

    # Chunk and save as datasets.DatasetDict
    def chunks(lst, n):
      """Yield successive n-sized chunks from lst."""
      for i in range(0, len(lst), n):
        yield lst[i:i + n]

    dataset_dict = {}
    for k, v in splits.items():
      if k == 'train' and crop_train == True:
        chunk_size = 2 * max_seq_length
      else:
        chunk_size = max_seq_length
      text = list(chunks(v, chunk_size))
      if drop_last and len(text[-1]) < chunk_size:
        text = text[:-1]
      dataset_dict[k] = datasets.Dataset.from_dict({'text': text})
    dataset = datasets.DatasetDict(dataset_dict)
    dataset.save_to_disk(cache_dir)
  else:
    dataset = datasets.load_from_disk(cache_dir)

  return dataset


def _group_texts(examples, block_size, bos, eos):
  # Concatenate all texts.
  concatenated_examples = list(itertools.chain(* examples['input_ids']))
  total_length = len(concatenated_examples)
  # TODO(yair): look into not dropping the remainder but rather padding it.
  # We drop the small remainder, and if the total_length < block_size - 2
  # we exclude this batch and return an empty dict.
  # We could add padding if the model supported it instead of
  # this drop, you can customize this part to your needs.
  new_block_size = block_size - 2  # [BOS] and [EOS] to be added
  total_length = (total_length // new_block_size) * new_block_size
  # Split by chunks of max_len.
  result = {}
  _values = []
  _attn_masks = []
  for i in range(0, total_length, new_block_size):
    _values.append(
      [bos]
      + concatenated_examples[i : i + new_block_size]
      + [eos])
    _attn_masks.append(torch.ones(block_size))
  result['input_ids'] = _values
  result['attention_mask'] = _attn_masks
  return result


def get_dataset(
    dataset_name, tokenizer, wrap, mode, cache_dir,
    block_size=1024, num_proc=len(os.sched_getaffinity(0)), streaming=False, data_files=None, use_endofjson_token=False):
  if wrap:
    filename = f'{dataset_name}_{mode}_bs{block_size}_wrapped.dat'
  else:
    filename = f'{dataset_name}_{mode}_bs{block_size}_unwrapped.dat'
  _path = os.path.join(cache_dir, filename)
  use_endofjson_token = use_endofjson_token
  if use_endofjson_token:
    LOGGER.info(f"Added <|endofjson|> token to tokenizer")
  
  if utils.fsspec_exists(_path):
    LOGGER.info(f'Loading data from: {_path}')
    return datasets.load_from_disk(_path).with_format('torch')
  LOGGER.info(f'Generating new data at: {_path}')

  crop_train = dataset_name == 'text8-crop'
  if mode == 'train' and crop_train:
    # double block size for sub-sampling
    block_size *= 2
  
  if dataset_name == 'wikitext103':
    dataset = datasets.load_dataset(
      'wikitext',
      name='wikitext-103-raw-v1',
      cache_dir=cache_dir)
  elif dataset_name == 'wikitext2':
    dataset = datasets.load_dataset(
      'wikitext',
      name='wikitext-2-raw-v1',
      cache_dir=cache_dir)
  elif dataset_name == 'ptb':
    dataset = datasets.load_dataset(
      'ptb_text_only', cache_dir=cache_dir)
  elif dataset_name == 'lambada':
    dataset = get_lambada_test_dataset()
  elif dataset_name == 'text8':
    assert wrap
    dataset = get_text8_dataset(
      cache_dir, max_seq_length=block_size)
  elif dataset_name == 'text8-crop':
    dataset = get_text8_dataset(
      cache_dir, max_seq_length=block_size, crop_train=True)
  elif dataset_name == 'openwebtext-train':
    dataset = datasets.load_dataset(
      'openwebtext',
      split='train[:-100000]',
      cache_dir=cache_dir,
      streaming=streaming)
  elif dataset_name == 'openwebtext-valid':
    dataset = datasets.load_dataset(
      'openwebtext',
      split='train[-100000:]',
      cache_dir=cache_dir,
      streaming=streaming)
  elif dataset_name == 'scientific_papers_arxiv':
    dataset = datasets.load_dataset(
      'scientific_papers', 'arxiv',
      trust_remote_code=True,
      cache_dir=cache_dir,
      streaming=streaming)
  elif dataset_name == 'scientific_papers_pubmed':
    dataset = datasets.load_dataset(
      'scientific_papers', 'pubmed',
      trust_remote_code=True,
      cache_dir=cache_dir,
      streaming=streaming)
  elif dataset_name == 'ag_news':
    dataset = datasets.load_dataset(
      'ag_news',
      cache_dir=cache_dir,
      streaming=streaming)
  elif dataset_name == 'schemabench':
    dataset = datasets.load_dataset(
      'json',
      data_files=data_files)
  else:
    dataset = datasets.load_dataset(
      dataset_name,
      cache_dir=cache_dir,
      streaming=streaming)

  if dataset_name in ['lambada', 'openwebtext-train',
                      'openwebtext-valid']:
    data = dataset
  else:
    data = dataset[mode]

  if dataset_name.startswith('wikitext'):
    detokenizer = wt_detokenizer
  elif dataset_name == 'ptb':
    detokenizer = ptb_detokenizer
  elif dataset_name == 'lm1b':
    detokenizer = lm1b_detokenizer
  elif dataset_name == 'lambada':
    detokenizer = lambada_detokenizer
  elif dataset_name.startswith('scientific_papers'):
    detokenizer = scientific_papers_detokenizer
  else:
    detokenizer = None

  def _apply_detokenizer(detokenizer):
    def detok(text):
      for i, t in enumerate(text, 0):
        text[i] = detokenizer(t)
      return text
    return detok
  
  EOS = tokenizer.encode(tokenizer.eos_token)[0]
  BOS = tokenizer.encode(tokenizer.bos_token)[0]

  def preprocess_and_tokenize(example):
    if dataset_name == 'ptb':
      text = example['sentence']
    elif 'scientific_papers' in dataset_name:
      text = example['article']
    else:
      text = example['text']
    
    if detokenizer is not None:
      text = _apply_detokenizer(detokenizer)(text)

    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'

    if wrap:
      tokens = tokenizer(text,
                         add_special_tokens=False,
                         return_attention_mask=False,
                         return_token_type_ids=False)
      tokens = {'input_ids':
                [t + [EOS] for t in tokens['input_ids']]}
      # Still missing BOS, but will be added in group_texts
    else:
      tokens = tokenizer(text,
                         max_length=block_size,
                         padding='max_length',
                         truncation=True,
                         add_special_tokens=True,
                         return_attention_mask=True,
                         return_token_type_ids=True)

    if dataset_name == 'schemabench':
      process_schemabench_with_endofjson(example, tokens, tokenizer, block_size)

    return tokens

  if streaming:
    tokenized_dataset = data.map(
      preprocess_and_tokenize,
      batched=True,
      desc='Tokenizing')
  else:
    tokenized_dataset = data.map(
      preprocess_and_tokenize,
      batched=True,
      num_proc=num_proc,
      load_from_cache_file=True,
      desc='Tokenizing')
  if dataset_name == 'ptb':
    tokenized_dataset = tokenized_dataset.remove_columns(
      'sentence')
  elif 'scientific_papers' in dataset_name:
    tokenized_dataset = tokenized_dataset.remove_columns([
      'article', 'abstract', 'section_names'])
  elif dataset_name == 'ag_news':
    tokenized_dataset = tokenized_dataset.remove_columns(
      ['text', 'label'])
  else:
    tokenized_dataset = tokenized_dataset.remove_columns(
      'text')

  if not wrap:
    tokenized_dataset.save_to_disk(_path)
    return tokenized_dataset.with_format('torch')

  group_texts = functools.partial(
    _group_texts, block_size=block_size, bos=BOS, eos=EOS)
  if streaming:
    chunked_dataset = tokenized_dataset.map(
      group_texts,
      batched=True,
      desc='Grouping')
  else:
    chunked_dataset = tokenized_dataset.map(
      group_texts,
      batched=True,
      num_proc=num_proc,
      load_from_cache_file=True,
      desc='Grouping')
    chunked_dataset.save_to_disk(_path)
  chunked_dataset = chunked_dataset.with_format('torch')
  return chunked_dataset


def _split_schemabench_dataset(train_set, config, valid_seed=None):
  """
  Split schemabench dataset into train and validation sets using configured ratio.
  
  Args:
    train_set: The full training dataset to split
    config: Configuration object containing train_valid_ratio
    valid_seed: Seed for reproducible splitting
    
  Returns:
    tuple: (train_subset, valid_subset)
  """
  split_ratio = float(getattr(config.data, "train_valid_ratio", 0.9))  # Default 0.9 -> train, 0.1 -> valid
  split_seed = valid_seed if valid_seed is not None else int(getattr(config, "seed", 42))
  
  # Calculate split sizes
  total_size = len(train_set)
  valid_portion = max(1, int(round(total_size * (1.0 - split_ratio))))
  train_portion = total_size - valid_portion
  
  # Create reproducible split
  generator = torch.Generator().manual_seed(split_seed)
  train_subset, valid_subset = torch.utils.data.random_split(
    train_set, [train_portion, valid_portion], generator=generator)
  
  return train_subset, valid_subset


def get_tokenizer(config):
  if config.data.tokenizer_name_or_path == 'text8':
    tokenizer = Text8Tokenizer()
  elif config.data.tokenizer_name_or_path == 'bert-base-uncased':
    tokenizer = transformers.BertTokenizer.\
      from_pretrained('bert-base-uncased')
  else:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
      config.data.tokenizer_name_or_path)

  if (isinstance(tokenizer, transformers.GPT2TokenizerFast)
      or isinstance(tokenizer, transformers.GPT2Tokenizer)):
    tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
      (tokenizer.bos_token, tokenizer.bos_token_id),
      (tokenizer.eos_token, tokenizer.eos_token_id))

  # For wrapped batches:
  #  [BOS] sent1 [EOS] sent2-fragment [EOS]
  #  [BOS] sent2-fragment [EOS] sent3 [EOS]
  if tokenizer.bos_token is None:
    if tokenizer.cls_token is None:
      raise AttributeError(
        'Tokenizer must have a bos_token or '
        f'cls_token: {tokenizer}')
    tokenizer.bos_token = tokenizer.cls_token
  if tokenizer.eos_token is None:
    if tokenizer.sep_token is None:
      raise AttributeError(
        'Tokenizer must have a eos_token '
        f'or sep_token: {tokenizer}')
    tokenizer.eos_token = tokenizer.sep_token
  if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

  return tokenizer
    

def get_dataloaders(config, tokenizer, skip_train=False,
                    skip_valid=False, valid_seed=None):
  num_gpus = torch.cuda.device_count()
  assert (config.loader.global_batch_size
          == (config.loader.batch_size
              * config.trainer.num_nodes
              * num_gpus
              * config.trainer.accumulate_grad_batches))
  if config.loader.global_batch_size % (
    num_gpus * config.trainer.accumulate_grad_batches) != 0:
    raise ValueError(
      f'Train Batch Size {config.training.batch_size}'
      f'not divisible by {num_gpus} gpus with accumulation '
      f'{config.trainer.accumulate_grad_batches}.')
  if config.loader.eval_global_batch_size % num_gpus != 0:
    raise ValueError(
      f'Eval Batch Size for {config.eval.batch_size} '
      f'not divisible by {num_gpus}.')

  data_files = getattr(config.data, 'data_files', None)
  if skip_train:
    train_set = None
  else:
    train_set = get_dataset(
      config.data.train,
      tokenizer,
      mode='train',
      wrap=config.data.wrap,
      cache_dir=config.data.cache_dir,
      block_size=config.model.length,
      data_files=data_files,
      use_endofjson_token=getattr(config, 'use_endofjson_token', False))
  
  if config.data.valid in ['text8', 'lm1b', 'ag_news']:
    validation_split = 'test'
  else:
    validation_split = 'validation'

  if config.data.train == 'schemabench':
    # Special case: schemabench splits the training set internally
    if skip_train:
      raise ValueError("Cannot skip training set for schemabench as it's needed for validation split")
    train_set, valid_set = _split_schemabench_dataset(train_set, config, valid_seed)
  else:
    if skip_valid:
      valid_set = None
    else:
      valid_set = get_dataset(
        config.data.valid,
        tokenizer,
        mode=validation_split,
        wrap=config.data.wrap,
        cache_dir=config.data.cache_dir,
        block_size=config.model.length,
        streaming=False,
        data_files=data_files,
        use_endofjson_token=getattr(config, 'use_endofjson_token', False))

  if skip_train:
    train_loader = None
  else:
    train_loader = torch.utils.data.DataLoader(
      train_set,
      batch_size=config.loader.batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=not config.data.streaming,
      persistent_workers=config.loader.num_workers > 0)
    train_loader.tokenizer = tokenizer
  if skip_valid:
    valid_loader = None
  else:
    if valid_seed is None:
      shuffle_valid = False
      generator = None
    else:
      shuffle_valid = True
      generator = torch.Generator().manual_seed(valid_seed)
    valid_loader = torch.utils.data.DataLoader(
      valid_set,
      batch_size=config.loader.eval_batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=shuffle_valid,
      generator=generator,
      persistent_workers=config.loader.num_workers > 0)
    # Will be used in generative perplexity calculation
    valid_loader.tokenizer = tokenizer

  return train_loader, valid_loader


# Samplers adapted from: https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/fault_tolerant_sampler.py


class RandomFaultTolerantSampler(torch.utils.data.RandomSampler):

  def __init__(self, *args, generator=None, **kwargs):
    # TD [2022-07-17]: We don't force the seed to be zero. We generate random seed,
    # which should be reproducible if pl.seed_everything was called beforehand.
    # This means that changing the seed of the experiment will also change the
    # sampling order.
    if generator is None:
      seed = int(torch.empty((), dtype=torch.int64).random_().item())
      generator = torch.Generator().manual_seed(seed)
    kwargs.pop('shuffle', None)
    super().__init__(*args, generator=generator, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'random_state': self.generator.get_state(),
            'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.generator.set_state(state_dict.get('random_state'))
    self.counter = state_dict['counter']
    # self.start_counter = self.counter
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.

  def __iter__(self) -> typing.Iterator[int]:
    n = len(self.data_source)

    self.state = self.generator.get_state()
    indices = torch.randperm(n, generator=self.generator).tolist()

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0


class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'epoch': self.epoch, 'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.epoch = state_dict['epoch']
    self.counter = state_dict['counter']
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.
  def __iter__(self):
    if self.shuffle:
      # deterministically shuffle based on epoch and seed
      g = torch.Generator()
      g.manual_seed(self.seed + self.epoch)
      indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
    else:
      indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

    if not self.drop_last:
      # add extra samples to make it evenly divisible
      padding_size = self.total_size - len(indices)
      if padding_size <= len(indices):
        indices += indices[:padding_size]
      else:
        indices += (indices * math.ceil(
          padding_size / len(indices)))[:padding_size]
    else:
      # remove tail of data to make it evenly divisible.
      indices = indices[:self.total_size]
    assert len(indices) == self.total_size

    # subsample
    indices = indices[self.rank:self.total_size:self.num_replicas]
    assert len(indices) == self.num_samples

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0


def process_json_with_endofjson_token(text, prompt, tokenizer, block_size, endofjson_token_id):
    """
    Process JSON text by inserting token ID 102 between structure tokens.
    
    Args:
        text: Full text containing prompt and response
        prompt: Prompt text
        tokenizer: Tokenizer to use
        block_size: Maximum sequence length (1024)
        endofjson_token_id: Token ID for <|endofjson|> (102)
        
    Returns:
        dict: Processed tokens with input_ids, attention_mask, and labels
    """
    # Tokenize full text and prompt
    text_tokens = tokenizer(text, 
                           max_length=block_size,
                           padding='max_length', 
                           truncation=True,
                           add_special_tokens=True)['input_ids']
    
    prompt_tokens = tokenizer(prompt, add_special_tokens=True)['input_ids']
    prompt_len = min(len(prompt_tokens), len(text_tokens)) - 1
    # prompt_mask = tokenizer.decode(torch.tensor(text_tokens) * (torch.arange(block_size) < prompt_len-1))
    
    # Get response tokens (everything after prompt, excluding padding)
    response_start_idx = prompt_len - 1  # 왜 그렇지?? <eos> 추가돼서 그런건가...
    response_tokens = [t for t in text_tokens[response_start_idx:] if t != tokenizer.pad_token_id]
    
    if not response_tokens:
        # No response tokens, return original
        return {
            'input_ids': text_tokens,
            'attention_mask': [1 if t != tokenizer.pad_token_id else 0 for t in text_tokens],
            'labels': text_tokens.copy()
        }
    
    # Get JSON structure mask for response tokens
    json_structure_mask = get_json_structure_mask(response_tokens)  # 처음 "\n"과 <|endoftext|> 도 포함.
    structure_indices = [i for i, is_struct in enumerate(json_structure_mask) if is_struct]
    
    if len(structure_indices) < 2:
        # Need at least 2 structure tokens to insert between them
        return {
            'input_ids': text_tokens,
            'attention_mask': [1 if t != tokenizer.pad_token_id else 0 for t in text_tokens],
            'labels': text_tokens.copy()
        }
    
    # Calculate available space and insertion count
    # Available space = max_seq_length - current_total_length
    current_total_length = prompt_len + len(response_tokens)
    available_space = block_size - current_total_length
    
    print(f"DEBUG: Prompt len: {prompt_len}")
    print(f"DEBUG: Response len: {len(response_tokens)}")  
    print(f"DEBUG: Current total: {current_total_length}")
    print(f"DEBUG: Block size: {block_size}")
    print(f"DEBUG: Available space: {available_space}")
    print(f"DEBUG: Structure indices: {structure_indices}")
    
    # Number of gaps between structure tokens
    num_gaps = len(structure_indices) - 1
    
    if available_space <= 0:
        # No space available, return original
        return {
            'input_ids': text_tokens,
            'attention_mask': [1 if t != tokenizer.pad_token_id else 0 for t in text_tokens],
            'labels': text_tokens.copy()
        }
    
    # Limit the number of tokens per gap to a reasonable amount  
    max_tokens_per_gap = min(5, available_space // max(1, num_gaps))  # Max 5 tokens per gap
    
    if num_gaps == 0:
        tokens_per_gap = min(max_tokens_per_gap, available_space)
    else:
        tokens_per_gap = min(max_tokens_per_gap, available_space // num_gaps)
    
    print(f"DEBUG: Num gaps: {num_gaps}")
    print(f"DEBUG: Available space: {available_space}")
    print(f"DEBUG: Max tokens per gap (limited): {max_tokens_per_gap}")
    print(f"DEBUG: Tokens per gap (final): {tokens_per_gap}")

    structure_indices = structure_indices[:-1]  # Exclude first and last structure tokens
    if tokens_per_gap == 0:
        # Insert tokens one by one in each gap until available_space is exhausted
        tokens_to_insert = available_space
        new_response_tokens = response_tokens[:]
        offset = 0
        
        # Keep inserting 1 token per gap cyclically until tokens_to_insert reaches 0
        while tokens_to_insert > 0:
            for i in range(1, len(structure_indices)):  # Start from second structure token
                if tokens_to_insert <= 0:
                    break
                    
                insertion_index = structure_indices[i] + offset
                # Insert 1 token at this gap
                new_response_tokens = (
                    new_response_tokens[:insertion_index] + 
                    [endofjson_token_id] + 
                    new_response_tokens[insertion_index:]
                )
                offset += 1
                tokens_to_insert -= 1
    else:
        # Insert calculated number of tokens before each structure token (except the first)
        new_response_tokens = response_tokens[:]
        offset = 0
        
        for i in range(1, len(structure_indices)):  # Start from second structure token
            insertion_index = structure_indices[i] + offset
            tokens_to_insert_batch = [endofjson_token_id] * tokens_per_gap
            
            new_response_tokens = (
                new_response_tokens[:insertion_index] + 
                tokens_to_insert_batch + 
                new_response_tokens[insertion_index:]
            )
            offset += tokens_per_gap
    
    # Build final token sequence
    new_tokens = text_tokens[:prompt_len] + new_response_tokens
    
    # Pad or truncate to block_size
    if len(new_tokens) > block_size:
        new_tokens = new_tokens[:block_size]
    else:
        # Pad with pad tokens
        new_tokens.extend([tokenizer.pad_token_id] * (block_size - len(new_tokens)))
    
    # Create attention mask
    attention_mask = [1 if t != tokenizer.pad_token_id else 0 for t in new_tokens]
    
    # Create labels (same as input_ids for language modeling)
    labels = new_tokens.copy()
    
    return {
        'input_ids': new_tokens,
        'attention_mask': attention_mask,
        'labels': labels
    }


def process_schemabench_with_endofjson(example, tokens, tokenizer, block_size):
    """
    Process schemabench dataset with <|endofjson|> token insertion.
    Uses existing unused token ID 102 instead of adding new token to avoid vocab size issues.
    """
    try:
        # Use existing unused token instead of adding new token
        # Token ID 102 ('�') is never used in the dataset according to analysis
        endofjson_token_id = 102
        LOGGER.info(f"Using existing unused token ID {endofjson_token_id} as <|endofjson|> replacement")
        
        # No vocab size changes needed since we're reusing existing token
        
        # Handle batched processing
        texts = example['text'] if isinstance(example['text'], list) else [example['text']]
        prompts = example['prompt'] if isinstance(example['prompt'], list) else [example['prompt']]
        
        if len(texts) != len(prompts):
            if len(prompts) == 1:
                prompts = prompts * len(texts)
            else:
                raise ValueError(f"Mismatch between texts ({len(texts)}) and prompts ({len(prompts)})")
        
        processed_input_ids = []
        processed_attention_masks = []
        prompt_masks = []
        json_structure_masks = []
        
        for text, prompt in zip(texts, prompts):
            # Process each example
            processed = process_json_with_endofjson_token(
                text, prompt, tokenizer, block_size, endofjson_token_id
            )
            
            processed_input_ids.append(processed['input_ids'])
            processed_attention_masks.append(processed['attention_mask'])
            
            # Generate masks for the processed tokens
            prompt_tokens = tokenizer(prompt, add_special_tokens=True)['input_ids']
            prompt_len = min(len(prompt_tokens), len(processed['input_ids'])) - 1
            
            # Create prompt mask
            prompt_mask = [True] * prompt_len + [False] * (len(processed['input_ids']) - prompt_len)
            prompt_masks.append(prompt_mask)
            
            # Create JSON structure mask for processed tokens
            json_structure_mask = [False] * len(processed['input_ids'])
            
            # Mark structure tokens
            for i, token_id in enumerate(processed['input_ids']):
                if i >= prompt_len:  # Only check response part
                    if token_id in set(JSON_STRUCTURE_TOKEN_IDS):
                        json_structure_mask[i] = True
            
            json_structure_masks.append(json_structure_mask)
        
        # Update tokens dictionary
        tokens['input_ids'] = processed_input_ids
        tokens['attention_mask'] = processed_attention_masks
        tokens['prompt_mask'] = prompt_masks
        tokens['json_structure_mask'] = json_structure_masks
        
    except Exception as e:
        LOGGER.error(f"Failed to process schemabench with endofjson: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to original processing
        # process_schemabench_masks(example, tokens, tokenizer, block_size)