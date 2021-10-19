from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers import Tokenizer
from tokenizers.models import WordPiece, BPE
from tokenizers.implementations import SentencePieceBPETokenizer
from glob import glob
from transformers import PreTrainedTokenizerFast

"""
# WHEN USING

from transformers import PreTrainedTokenizerFast

transformer_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer
)
"""
def main(tokenizer_type='sentencepiece_bpe', vocab_size=9000):
    files = glob('*.txt')
    print('tokenize', files)
    trainer_kwargs = {'special_tokens': ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], 'vocab_size': vocab_size, 'min_frequency':0}

    if tokenizer_type == 'sentencepiece_bpe':
        tokenizer = SentencePieceBPETokenizer(dropout=0.9999)
        tokenizer.train(files=files, **trainer_kwargs)
    elif tokenizer_type == 'bpe':
        tokenizer = Tokenizer(BPE(dropout=0.9999))
        trainer = BpeTrainer(files=files, **trainer_kwargs)
        tokenizer.train(files=files, trainer=trainer)
    elif tokenizer_type == 'wordpiece':
        tokenizer = Tokenizer(WordPiece())
        trainer = WordPieceTrainer(**trainer_kwargs)
        tokenizer.train(files=files, trainer=trainer)
    tokenizer.save(f"tokenizer-{tokenizer_type}.json")

if __name__ == '__main__':
    main()