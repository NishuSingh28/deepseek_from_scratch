"""
Tokenizer training and utilities using SentencePiece.
"""

import sentencepiece as spm
from pathlib import Path


def train_tokenizer(input_file, model_prefix, vocab_size=32000):
    """Train a SentencePiece tokenizer."""
    print(f"Training tokenizer on {input_file}...")
    print(f"Vocab size: {vocab_size}")
    
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece='[PAD]',
        unk_piece='[UNK]',
        bos_piece='[BOS]',
        eos_piece='[EOS]',
        character_coverage=1.0,
        byte_fallback=True,
    )
    
    print(f"✅ Tokenizer saved as {model_prefix}.model")


class Tokenizer:
    """Wrapper for SentencePiece tokenizer."""
    
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(model_path))
        
        self.pad_id = self.sp.pad_id()
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        
        print(f"✅ Tokenizer loaded from {model_path}")
        print(f"   Vocab size: {self.vocab_size}")
    
    @property
    def vocab_size(self):
        return self.sp.get_piece_size()
    
    def encode(self, text, add_bos=False, add_eos=False):
        ids = self.sp.encode(text)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids
    
    def decode(self, ids):
        return self.sp.decode(ids)
