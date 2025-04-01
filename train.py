# import own files
from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset # huggingface allow easy download
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


import warnings

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
  sos_idx = tokenizer_tgt.token_to_id('[SOS]')
  eos_idx = tokenizer_tgt.token_to_id('[EOS]')

  # precompute the encoder output and reuse it for every token we get from the decoder
  encoder_output = model.encode(source, source_mask)

  # initialize the decoder input with the sos token
  decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)

  while True:
    if decoder_input.size(1) == max_len:
      break

    # build mask for the target (decoder input)
    decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

    # calculate the output of the decoder
    out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

    # get the max token
    prob = model.project(out[:,1])

    #select the token with the max probability (because it is a greedy search)
    _, next_word = torch.max(prob, dim=1)

    decoder_input = torch.cat(
           [decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1
    )
    if next_word == eos_idx:
      break

  # remove the batch dim from the output which is decoder_input
  return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, write, num_examples=2):
  model.eval()
  count = 0
  console_width = 80


  with torch.no_grad():
    for batch in validation_ds:
      count += 1
      encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
      encoder_mask = batch['encoder_mask'].to(device) # (b, 1, 1, seq_len)

      assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
      model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

      source_text = batch['src_text'][0]
      target_text = batch['tgt_text'][0]
      model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

      # print to console
      print_msg('-'*console_width)
      print_msg(f'SOURCE: {source_text}')
      print_msg(f'TARGET: {target_text}')
      print_msg(f'PREDICTED: {model_out_text}')

      if count == num_examples:
        break

def get_all_sentences(ds, lang):
  # Generator function to extract sentences from a dataset for a specified language.
  for item in ds:
    yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    """
    Retrieves an existing tokenizer or builds a new one for a specified language.

    Parameters:
        - config (Dict): Configuration dictionary containing tokenizer_file path.
        - dataset (List[Dict]): List of dictionary entries representing the dataset.
        - lang (str): Language key indicating the target language.

    Returns:
        - tokenizer (Tokenizer): Tokenizer object for the specified language.
    """

    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    if not Path.exists(tokenizer_path):
      # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
      tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
      tokenizer.pre_tokenizer = Whitespace()
      trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
      tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
      tokenizer.save(str(tokenizer_path))
    else:
      tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_ds(config):
  # It only has the train split, so we divide it overselves
  ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')
  #ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

  # opus_books: 'en-it' (english to Italian) - 1st entry
  # {'id': '0', 'translation': {'en': 'Source: Project Gutenberg', 'it': 'Source: www.liberliber.it/Audiobook available here'}}

  # build tokenizer
  tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
  tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

  # keep train:validation = 0.9:0.1
  train_ds_size = int(0.9 * len(ds_raw))
  val_ds_size = len(ds_raw)-train_ds_size

  train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

  train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
  val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

  max_len_src = 0
  max_len_tgt = 0

  for item in ds_raw:
    src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
    tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
    max_len_src = max(max_len_src, len(src_ids))
    max_len_tgt = max(max_len_tgt, len(tgt_ids))

  print(f'Max length of source sentence: {max_len_src}')
  print(f'Max length of target sentence: {max_len_tgt}')

  train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
  val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

  return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
  model = build_transformer(vocab_src_len, vocab_tgt_len, config ['seq_len'], config ['seq_len'], config['d_model'])
  return model

def train_model(config):
  device = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')
  device = torch.device(device)
  print(f'Using  device: {device}')

  # Make sure the weights folder exists
  Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

  train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
  model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

  # tensorboard
  writer = SummaryWriter(config['experiment_name'])

  optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

  # store the state of model & optimizer, if user specified
  initial_epoch = 0
  global_step = 0

  preload = config['preload']
  model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
  if model_filename:
    #model_filename = get_weights_file_path(config, config['preload'])
    print(f'Preloading model {model_filename}')
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])
    initial_epoch = state['epoch'] + 1
    optimizer.load_state_dict(state['optimizer_state_dict'])
  else:
    print('No model to preload, starting from scratch')

  # label_smoothing=0.1 => take 10% from highest probablity and give it to others
  loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

  for epoch in range(initial_epoch, config['num_epochs']):
    batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
    for batch in batch_iterator:
      model.train()
      encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
      decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
      encoder_mask = batch['encoder_mask'].to(device)   # (B, 1, 1, seq_len)
      decoder_mask = batch['decoder_mask'].to(device)   # (B, 1, seq_len. seq_len)

      encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
      decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
      proj_output = model.projection_layer(decoder_output) # (B, seq_len, tgt_vocab_size)
      label = batch['label'].to(device) # (B, seq_len)

      # Compute the loss using a simple cross entropy
      # (B, seq_len, tgt_vocab_size) -> (B * seq_len, tgt_vocab_size)
      loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

      batch_iterator.set_postfix({'loss': f'{loss.item():6.3f}'})

      # log to tensorboard
      writer.add_scalar('train_loss', loss.item(), global_step)
      writer.flush()

      # backprop
      loss.backward()

      # update weights
      optimizer.step()
      optimizer.zero_grad() # optimizer.zero_grad(set_to_none=True)

      global_step += 1 # used for tensorboard
    run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt,config['seq_len'], device, lambda msg: batch_iterator.write(msg), writer, global_step)

    # save model at the end of every epoch
    model_filename = get_weights_file_path(config, f'{epoch:02d}')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step
    }, model_filename)

# Start training
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
