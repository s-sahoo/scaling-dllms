from abc import ABC, abstractmethod
import base64
import collections
import glob
import json
import os
import pathlib
import typing
import random
import struct

import numpy as np
import sentencepiece
import tiktoken
import torch

PATTERN_TIKTOKEN_V2 = "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"


class TinyLLamaTokenizer:
  def __init__(self, model_path) -> None:
    # some checkpoints have both files, `.model` takes precedence
    if os.path.isfile(model_path):
      self.processor = sentencepiece.SentencePieceProcessor(
        model_file=str(model_path))
      self.bos_token_id = self.processor.bos_id()
      self.eos_token_id  = self.processor.eos_id()
      self._vocab_size = self.processor.vocab_size()
      # add a mask token to the vocabulary
      self.mask_token_id = self._vocab_size
      self.mask_token = '[MASK]'
      self._vocab_size += 1
    else:
      raise NotImplementedError(model_path)

  @property
  def vocab_size(self) -> int:
    return self._vocab_size

  def __len__(self) -> int:
    return self._vocab_size

  def token_to_id(self, token: str) -> int:
    id_ = self.processor.piece_to_id(token)
    if id_ is None:
      raise ValueError(
        f"token {token!r} not found in the collection.")
    return id_

  def encode(self,
             string:str,
             device:typing.Optional[torch.device]=None,
             bos:bool=False,
             eos:bool=True,
             max_length:int=-1) -> torch.Tensor:
    tokens = self.processor.encode(string)
    if bos:
      bos_id = self.bos_token_id
      if bos_id is None:
        raise NotImplementedError(
          'This tokenizer does not defined a bos token')
      tokens = [bos_id] + tokens
    if eos:
      tokens = tokens + [self.eos_token_id]
    if max_length > 0:
      tokens = tokens[:max_length]
    return torch.tensor(tokens, dtype=torch.int,
                        device=device)

  def batch_encode(self, *args, **kwargs):
    return self.encode(*args, **kwargs)

  def decode(self, tensor:torch.Tensor) -> str:
    if tensor.ndim == 0:
      tokens = [tensor.item()]
    else:
      tokens = tensor.tolist()
    return self.processor.decode(tokens)

  def batch_decode(self, *args, **kwargs):
    return self.decode(*args, **kwargs)


class DatasetIterator:
  HDR_MAGIC = b"LITPKDS"
  HDR_SIZE = 24  # bytes
  DTYPES = {1: np.uint8, 2: np.int8, 3: np.int16,
            4: np.int32, 5: np.int64, 6: np.float32,
            7: np.float64, 8: np.uint16}
  
  def __init__(self, filenames, n_chunks, block_size,
               seed, shuffle, wrap, train_start_file_idx=0):
    self._seed = seed
    self._shuffle = shuffle
    self._rng = np.random.default_rng(seed) if shuffle else None
    self._block_idxs = None

    self._wrap = wrap
    self._filenames = filenames
    self._file_idx = train_start_file_idx

    self._n_chunks = n_chunks

    self._dtype = None
    self._block_size = block_size
    self._n_blocks = None

    self._mmaps = []
    self._buffers = []
    self._block_idxs = []
    self._curr_idx = 0

    self._load_n_chunks()

  def _read_header(self, path):
    with open(path, "rb") as f:
      magic = f.read(len(self.HDR_MAGIC))
      assert magic == self.HDR_MAGIC, (
        "File doesn't match expected format.")
      version = struct.unpack("<Q", f.read(8))
      assert version == (1,)
      (dtype_code,) = struct.unpack("<B", f.read(1))
      dtype = self.DTYPES[dtype_code]
      (chunk_size,) = struct.unpack("<Q", f.read(8))
    return dtype, chunk_size

  def _close_mmaps(self):
    for mmap in self._mmaps:
      mmap._mmap.close()

  def _load_n_chunks(self):
    self._close_mmaps()
    self._mmaps = []
    self._buffers = []
    if self._n_chunks > len(self._filenames[self._file_idx :]):
      self._file_idx = 0
    for i in range(self._n_chunks):
      filename = self._filenames[self._file_idx + i]
      if self._dtype is None:
        self._dtype, self._chunk_size = self._read_header(filename)
        self._n_blocks = self._chunk_size // self._block_size
      mmap = np.memmap(filename, mode='r', order='C',
                       offset=self.HDR_SIZE)
      self._mmaps.append(mmap)
      self._buffers.append(memoryview(mmap))

    self._file_idx += self._n_chunks
    n_all_blocks = self._n_chunks * self._n_blocks
    if self._shuffle:
      self._block_idxs = self._rng.permutation(n_all_blocks)
    else:
      self._block_idxs = range(n_all_blocks)
    self._curr_idx = 0

  def __del__(self):
    self._close_mmaps()
    del self._mmaps
    del self._buffers

  def __iter__(self):
      return self

  def __next__(self):
    if self._curr_idx >= len(self._block_idxs):
      self._load_n_chunks()
    block_idx = self._block_idxs[self._curr_idx]
    chunk_id = block_idx // self._n_blocks
    buffer = self._buffers[chunk_id]
    elem_id = (block_idx % self._n_blocks) * self._block_size
    offset = np.dtype(self._dtype).itemsize * elem_id
    arr = np.frombuffer(buffer,
                        dtype=self._dtype,
                        count=self._block_size,
                        offset=offset)
    self._curr_idx += 1
    return {
      'input_ids': torch.from_numpy(arr.astype(np.int64)),
      'file_idx': self._file_idx,
      'curr_idx': self._curr_idx}


class CustomDataset(torch.utils.data.IterableDataset):
  def __init__(self, filenames, n_chunks, block_size,
               seed=0, shuffle=True, wrap=False,
               train_start_file_idx=0):
    self._filenames = filenames
    self._n_chunks = n_chunks
    self._block_size = block_size 
    self._seed = seed
    self._shuffle = shuffle
    self._wrap = wrap
    self._train_start_file_idx = train_start_file_idx

  def __iter__(self):
    return DatasetIterator(
      filenames=self._filenames,
      n_chunks=self._n_chunks,
      block_size=self._block_size,
      seed=self._seed,
      shuffle=self._shuffle,
      wrap=self._wrap,
      train_start_file_idx=self._train_start_file_idx)


def create_dataloader(
    batch_size:int,
    block_size:int,
    filenames:list,
    n_chunks:int=8,
    shuffle:bool=True,
    seed:int=12345,
    pin_memory=True,
    num_workers=1,
    train_start_file_idx=0):
  random.seed(seed)
  random.shuffle(filenames)

  dataset = CustomDataset(
    filenames,
    n_chunks=n_chunks,
    block_size=block_size,
    shuffle=shuffle,
    seed=seed,
    train_start_file_idx=train_start_file_idx)

  return torch.utils.data.DataLoader(
    dataset, 
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=True)


def create_dataloaders(config, seed, train_start_file_idx=0,
                       current_device=0, total_devices=1):
  data_dir = pathlib.Path(config.data.cache_dir)
  if config.data.train == 'slim_pajama':
    train_filenames = sorted(glob.glob(str(data_dir / 'train*')))
    val_filenames = sorted(glob.glob(str(data_dir / 'validation*')))
  elif config.data.train == 'nvidia':
    if config.data.mode == 'phase1':
      # data_dir/train/ and data_dir/validation/ contain
      # filenames with the pattern [0-255]_*.bin
      assert current_device < total_devices
      train_filenames = []
      bucket_size = 256 // total_devices
      assert bucket_size * total_devices == 256
      for i in range(bucket_size):
        idx = bucket_size * current_device + i
        train_filenames.extend(
          glob.glob(str(data_dir / f'train/{idx}*.bin')))
      train_filenames = sorted(train_filenames)
      val_filenames = sorted(
        glob.glob(str(data_dir / 'validation/*')))
    elif config.data.mode == 'phase2':
      filter = f'shard_{current_device}/*.bin'
      train_filenames = sorted(
        glob.glob(str(data_dir / 'train' / filter)))
      val_filenames = sorted(
        glob.glob(str(data_dir / 'validation' / filter)))
  train_dataloader = create_dataloader(
    batch_size=config.loader.batch_size,
    block_size=config.model.length,
    filenames=train_filenames,
    n_chunks=config.loader.n_chunks,
    shuffle=True,
    seed=seed,
    num_workers=1,
    pin_memory=config.loader.pin_memory,
    train_start_file_idx=train_start_file_idx)
  val_dataloader = create_dataloader(
    batch_size=config.loader.eval_batch_size,
    block_size=config.model.length,
    filenames=val_filenames,
    n_chunks=config.loader.n_chunks,
    shuffle=False,
    seed=seed,
    num_workers=1,
    pin_memory=config.loader.pin_memory)
  return train_dataloader, val_dataloader



class MegatronTokenizer(ABC):
  """Abstract class for tokenizer

  Absent a config or class-specific tracking of which objects are uniquely identifying, we must
  include all key word arguments as unique identifiers

  Args:
      tokenizer_paths (Tuple[str]): All tokenizer source paths or prefixes

      tokenizer_options (Dict[str, Any]): All tokenizer options
  """

  def __init__(self, *tokenizer_paths, **tokenizer_options):
    self.unique_identifiers = collections.OrderedDict()
    self.unique_identifiers["class"] = type(self).__name__
    self.unique_identifiers["tokenizer_path"] = list(tokenizer_paths)
    for option in tokenizer_options:
        self.unique_identifiers[option] = str(tokenizer_options[option])

    self.unique_description = json.dumps(self.unique_identifiers, indent=4)

    super().__init__()

  @abstractmethod
  def tokenize(self, text):
    """Convert text to embedding ids

    Args:
        text (str): The text to convert

    Returns:
        numpy.ndarray: The converted embedding ids
    """
    pass

  def detokenize(self, ids):
    """Convert embedding ids to text

    Args:
        ids (numpy.ndarray): The ids to convert

    Returns:
        str: The converted text

    Raises:
        NotImplementedError: Non-abstract, optional method
    """
    raise NotImplementedError(
      f'{type(self).__name__} has no method "detokenize"')

  def offsets(self, ids, text):
    """Convert embedding ids to text offsets

    Args:
        ids (list[int]): The ids to convert
        text (str): The text to convert

    Returns:
        list[int]: The converted offsets

    Raises:
        NotImplementedError: Non-abstract, optional method
    """
    raise NotImplementedError(
      f'{type(self).__name__} has no method "offsets"')

  @property
  @abstractmethod
  def vocab(self):
    """Dictionary from vocab text token to id token"""
    pass

  @property
  @abstractmethod
  def inv_vocab(self):
    """Dictionary from vocab id token to text token"""
    pass

  @property
  @abstractmethod
  def vocab_size(self):
    """The vocabulary size"""
    pass

  @property
  def cls(self):
    """The CLS token id

    Raises:
        NotImplementedError: Non-abstract, optional attribute
    """
    raise NotImplementedError(
      f'{type(self).__name__} has no attribute "cls"')

  @property
  def sep(self):
    """The SEP token id

    Raises:
        NotImplementedError: Non-abstract, optional attribute
    """
    raise NotImplementedError(
      f'{type(self).__name__} has no attribute "sep"')

  @property
  def pad(self):
    """The PAD token id

    Raises:
        NotImplementedError: Non-abstract, optional attribute
    """
    raise NotImplementedError(
      f'{type(self).__name__} has no attribute "pad"')

  @property
  def eod(self):
    """The EOD token id

    Raises:
        NotImplementedError: Non-abstract, optional attribute
    """
    raise NotImplementedError(
      f'{type(self).__name__} has no attribute "eod"')

  @property
  def bos(self):
    """The BOS token id

    Raises:
        NotImplementedError: Non-abstract, optional attribute
    """
    raise NotImplementedError(
      f'{type(self).__name__} has no attribute "bos"')

  @property
  def eos(self):
    """The EOS token id

    Raises:
        NotImplementedError: Non-abstract, optional attribute
    """
    raise NotImplementedError(
      f'{type(self).__name__} has no attribute "eos"')

  @property
  def mask(self):
    """The MASK token id

    Raises:
        NotImplementedError: Non-abstract, optional attribute
    """
    raise NotImplementedError(
      f'{type(self).__name__} has no attribute "mask"')


def reload_mergeable_ranks(path,  max_vocab=None):
  """
    Reloads a tokenizer JSON file and converts it to Tiktoken format.
  """
  assert path.endswith('.json')

  # reload vocab
  with open(path, 'r') as f:
    vocab = json.load(f)
  assert isinstance(vocab, list)
  if max_vocab is not None:
    vocab = vocab[:max_vocab]

  # build ranks
  ranks: typing.Dict[bytes, int] = {}
  for i, x in enumerate(vocab):
    assert x.keys() == {'rank', 'token_bytes',
                        'token_str'}
    assert x['rank'] == i
    merge = base64.b64decode(x['token_bytes'])
    assert i >= 256 or merge == bytes([i])
    ranks[merge] = x['rank']

  # sanity check
  assert len(ranks) == len(vocab)
  assert set(ranks.values()) == set(range(len(ranks)))

  return ranks


class CustomTikTokenizer(MegatronTokenizer):
  SPECIAL_TOKENS = ['<unk>', '<s>', '</s>', '<mask>']

  def __init__(self, path, pattern, vocab_size,
               num_special_tokens, special_tokens):
    super().__init__(
      path,
      pattern=pattern,
      vocab_size=vocab_size,
      num_special_tokens=num_special_tokens,
      special_tokens=special_tokens)

    if vocab_size is None:
      vocab_size = 2**17  # Fallback vocab size is 131072.
    self._vocab_size = vocab_size

    
    if special_tokens is None:
      special_tokens = self.SPECIAL_TOKENS.copy()
    assert len(special_tokens) == len(set(special_tokens)), (
      f'Special tokens should be unique: {special_tokens}')
    assert (len(special_tokens) <= num_special_tokens
            < self._vocab_size)
    assert set(self.SPECIAL_TOKENS) <= set(special_tokens), (
      f'Custom special tokens should include {self.SPECIAL_TOKENS}')

    special_filler = [
      f'<SPECIAL_{i}>' for i in range(
        len(special_tokens), num_special_tokens)]
    special_tokens = special_tokens + special_filler
    assert (
      len(set(special_tokens)) == len(special_tokens)
        == num_special_tokens), (
          f'Special tokens should be unique: {special_tokens}')
    inner_vocab_size = self._vocab_size - num_special_tokens

    token_to_id_sans_special = reload_mergeable_ranks(
        path, max_vocab=inner_vocab_size)
    # Create space for special tokens.
    token_to_id_sans_special = {
      t: i + num_special_tokens
        for t, i in token_to_id_sans_special.items()}

    special_tokens = {
      t: i for i, t in enumerate(special_tokens)}
    self._unk_id = special_tokens['<unk>']
    self._bos_id = special_tokens['<s>']
    self._eos_id = special_tokens['</s>']
    self._mask_id = special_tokens['<mask>']
    
    # Public attributes for backward compatibility.
    self.mask_token = '<mask>'
    self.mask_token_id = self._mask_id

    # Create tiktoken model.
    self._model = tiktoken.Encoding(
        name=pathlib.Path(path).parent.name,
        pat_str=pattern,
        mergeable_ranks=token_to_id_sans_special,
        special_tokens=special_tokens)

    # Create final _id_to_token and _token_to_id data 
    # structures with special tokens inserted
    # at appropriate locations.
    assert set(
      token_to_id_sans_special.keys()).isdisjoint(
        set(special_tokens.keys()))
    self._token_to_id = token_to_id_sans_special.copy()
    self._token_to_id.update(special_tokens)
    self._id_to_token = {
      v: k for k, v in self._token_to_id.items()}
    assert (set(range(self._vocab_size))
            == set(self._id_to_token.keys()))

  @property
  def bos(self) -> int:
    return self._bos_id

  @property
  def eos(self) -> int:
    return self._eos_id

  @property
  def unk(self) -> int:
    return self._unk_id

  @property
  def mask(self) -> int:
    return self._mask_id

  @property
  def eod(self) -> int:
    return self._eos_id

  @property
  def vocab(self):
    return self._token_to_id

  @property
  def inv_vocab(self):
    return self._id_to_token

  def tokenize(self, s, bos=False, eos=False):
    tokens = self._model.encode_ordinary(s)
    if bos:
      tokens = [self.bos, *tokens]
    if eos:
      tokens = [*tokens, self.eos]

    return tokens

  def detokenize(self, tokens):
    return self._model.decode(tokens)

  def offsets(self, ids, text):
    try:
      return self._model.decode_with_offsets(ids)[1]
    except UnicodeDecodeError:
      # Tiktoken has an unnecessary check that raises UnicodeDecodeError
      # from `text = b"".join(token_bytes).decode("utf-8", errors="strict")`
      # which is not needed for our use case. So we re-implement it, without
      # the check.

      token_bytes = self._model.decode_tokens_bytes(ids)
      text_len = 0
      offsets = []
      for token in token_bytes:
        offsets.append(max(0, text_len - (0x80 <= token[0] < 0xC0)))
        text_len += sum(1 for c in token if not 0x80 <= c < 0xC0)
      return offsets

  @property
  def vocab_size(self):
      return self._vocab_size

  @property
  def encoder(self):
      return self._token_to_id

  @property
  def decoder(self):
      return self._id_to_token

  def __len__(self):
    return self._vocab_size

  def decode(self, tokens):
    return self.detokenize(tokens.cpu().numpy().reshape(-1))

  def batch_decode(self, tokens):
    tokens = tokens.cpu().numpy()
    return [self.detokenize(token) for token in tokens]


def get_tokenizer(config):
  if 'tinyllama' in config.data.tokenizer_name_or_path:
    return TinyLLamaTokenizer(
      config.data.tokenizer_name_or_path)
  return CustomTikTokenizer(
      path=config.data.tokenizer_name_or_path,
      pattern=PATTERN_TIKTOKEN_V2,
      vocab_size=None,
      num_special_tokens=1000,
      special_tokens=None)
