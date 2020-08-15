#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import regex


def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings.
    We specifically avoids mapping to whitespace/control characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    """
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class GPT2Tokenizer(object):

    def __init__(
            self,
            vocab_file,
            merges_file,
            errors="replace",
            bos_token="<s>",
            eos_token="</s>",
            sep_token="</s>",
            cls_token="<s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",):

        self.max_len = 1024
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.sep_token = sep_token
        self.cls_token = cls_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.mask_token = mask_token

        # load vocabulary
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)  # word to id
        self.decoder = {v: k for k, v in self.encoder.items()}  # id to word

        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        self.bos_token_id = self._convert_token_to_id(bos_token)
        self.eos_token_id = self._convert_token_to_id(eos_token)
        self.sep_token_id = self._convert_token_to_id(sep_token)
        self.cls_token_id = self._convert_token_to_id(cls_token)
        self.unk_token_id = self._convert_token_to_id(unk_token)
        self.pad_token_id = self._convert_token_to_id(pad_token)
        self.mask_token_id = self._convert_token_to_id(mask_token)

        self.begin_token = '<s>'
        self.end_token = '</s>'
        self.sep_token = '</s>'  # https://github.com/pytorch/fairseq/blob/master/fairseq/models/bart/hub_interface.py#L74
        self.has_space_in_subword = True  # GPT2 tokenization contains space

    def vocab_size(self):
        return len(self.encoder)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def _tokenize(self, text, add_prefix_space=False):
        """ Tokenize a string.
            Args:
                - add_prefix_space (boolean, default False):
                    Begin the sentence with at least one space to get invariance to word order in GPT-2 (and RoBERTa) tokenizers.
        """
        if add_prefix_space:
            text = " " + text.strip()

        bpe_tokens = []
        for token in regex.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding controle tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    def convert_tokens_to_ids(self, tokens):
        return [self._convert_token_to_id(token) for token in tokens]

    def encode(self, sentence, *addl_sentences, max_len=1024):
        tokens = self._tokenize(sentence, add_prefix_space=True)
        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]
        tokens.insert(0, '<s>')
        for s in addl_sentences:
            cur_tokens = self._tokenize(s, add_prefix_space=True)
            if len(cur_tokens) > max_len - 2:
                cur_tokens = cur_tokens[:max_len - 2]
            cur_tokens.insert(0, '</s>')
            tokens.extend(cur_tokens)
        tokens.append('</s>')
        token_ids = self.convert_tokens_to_ids(tokens)
        return tokens, token_ids

    def decode(self, token_ids):
        if token_ids[0] == self.bos_token_id:
            token_ids = token_ids[1:]  # remove <s>
        tokens = []
        for token_id in token_ids:
            if token_id in [self.bos_token_id, self.pad_token_id, self.eos_token_id]:
                continue
            tokens.append(self._convert_id_to_token(token_id))
        sentence = " ".join(tokens).replace(' ', '').replace('Ġ', ' ').strip()
        return sentence

    def tokenize(self, token):
        return self._tokenize(token, add_prefix_space=True)

    def map_subtokens_to_ids(self, sub_tokens):
        return self.convert_tokens_to_ids(sub_tokens)


if __name__ == '__main__':
    vocab_file = "/Users/zhiguow/Downloads/pretrained_bart_model/roberta-large-vocab.json"
    merges_file = "/Users/zhiguow/Downloads/pretrained_bart_model/roberta-large-merges.txt"
    tokenizer = GPT2Tokenizer(vocab_file, merges_file)
    sample = "Hello world"
    sample = "Fine-tuning on CNN-DM <ANS> summarization task"
    tokens, token_ids = tokenizer.encode(sample)
    # tokens = tokenizer._tokenize("Hello world", add_prefix_space=False)
    # ids = tokenizer.convert_tokens_to_ids(tokens)
    print("tokens:", tokens)
    print("token_ids:", token_ids)
    print("decding result:", tokenizer.decode(token_ids))

    print('DONE!')