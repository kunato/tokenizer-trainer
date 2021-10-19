from tokenizers.trainers import WordPieceTrainer
from tokenizers import Tokenizer, SentencePieceBPETokenizer
from tokenizers.models import WordPiece, BPE
from tokenizers.implementations import SentencePieceBPETokenizer
from glob import glob


def main(tokenizer_type='sentencepiece_bpe'):
    files = glob('*.txt')
    print('tokenize', files)
    if tokenizer_type == 'sentencepiece_bpe':
        tokenizer = SentencePieceBPETokenizer(dropout=0.9999)
    elif tokenizer_type == 'bpe':
        tokenizer = Tokenizer(BPE(dropout=0.9999))
    elif tokenizer_type == 'wordpiece':
        tokenizer = Tokenizer(WordPiece())
    trainer = WordPieceTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=9000, min_frequency=0)
    tokenizer.train(files=files, trainer=trainer)
    tokenizer.save("tokenizer-wiki.json")

if __name__ == '__main__':
    main()