import unicodedata
import torch
import numpy as np

#-\'.ü!"#%&()*+,/:;?
Alphabets = {
    #'!#&():;?*%'
    'all': '` ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789\'-"/,.+_!#&():;?',# n_class: 80
    'iam_word': '` ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789\'-"/,.+_!#&():;?', # n_class: 80
    'iam_line': '` ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789\'-"/,.+_!#&():;?', # n_class: 80
    'cvl_word': '` ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789\'-"/,.+_!#&():;?', # n_class: 80
    'custom': '` ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789\'-"/,.+_!#&():;?',
    # 'cvl_word': '` ABDEFGHILNPRSTUVWYZabcdefghiklmnopqrstuvwxyz\'-_159', # n_class: 52
    'rimes_word': '` ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789%\'-/Éàâçèéêëîïôùû' # n_class: 81
}


def get_alphabet_size(alphabet_key):
    """Return the number of character classes (including blank) for a given alphabet key."""
    return len(Alphabets[alphabet_key])


class strLabelConverter(object):
    """Convert between str and label.
    NOTE:
        Insert `blank` to the alphabet for CTC.
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet_key, ignore_case=False):
        if isinstance(alphabet_key, str) and alphabet_key in Alphabets:
            alphabet = Alphabets[alphabet_key]
        else:
            # Allow passing a raw alphabet string directly for custom languages
            alphabet = alphabet_key
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet
        self.n_class = len(alphabet)

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i

    def encode(self, text, max_len=None):
        """Support batch or single str.
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            text = [
                self.dict.get(char.lower() if self._ignore_case else char, 0)
                for char in text
            ]
            return text

        length = []
        result = []
        results = []
        for item in text:
            length.append(len(item))
            for char in item:
                index = self.dict.get(char, 0)
                result.append(index)
            results.append(result)
            result = []

        labels = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(seq) for seq in results], batch_first=True)
        lengths = torch.IntTensor(length)

        if max_len is not None and max_len > labels.size(-1):
            pad_labels = torch.zeros((labels.size(0), max_len)).long()
            pad_labels[:, :labels.size(-1)] = labels
            labels = pad_labels

        return labels, lengths

    def decode(self, t, length=None, raw=False):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
        """
        def nonzero_count(x):
            return len(x.nonzero(as_tuple=False))

        if isinstance(t, list):
            t = torch.IntTensor(t)
            length = torch.IntTensor([len(t)])
        elif length is None:
            length = torch.IntTensor([nonzero_count(t)])

        if length.numel() == 1:
            length = length[0]
            if raw:
                t_indices = t.tolist() if isinstance(t, torch.Tensor) else list(t)
                return ''.join([self.alphabet[i] for i in t_indices if 0 <= i < len(self.alphabet)])
            else:
                char_list = []
                if t.dim() == 2:
                    t = t[0]
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        idx = t[i].item() if hasattr(t[i], 'item') else int(t[i])
                        if 0 <= idx < len(self.alphabet):
                            char_list.append(self.alphabet[idx])
                return ''.join(char_list)
        else:
            # batch mode
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[i, :l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


def get_true_alphabet(name):
    tag = '_'.join(name.split('_')[:2])
    if tag in Alphabets:
        return Alphabets[tag]
    # Fallback: try just the first token
    first = name.split('_')[0]
    for key in Alphabets:
        if key.startswith(first):
            return Alphabets[key]
    return Alphabets.get('all', '')


def get_lexicon(path, true_alphabet, max_length=20, ignore_case=True):
    words = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) < 2:
                    continue

                word = ''.join(ch for ch in line if ch in true_alphabet)
                if len(word) != len(line) or len(word) >= max_length:
                    continue
                if ignore_case:
                    word = word.lower()
                words.append(word)
    except FileNotFoundError as e:
        print(e)
    return words


def word_capitalize(word):
    """Capitalize the first character of a word in a Unicode-safe way.
    
    Works correctly with any script (Latin, Cyrillic, Arabic, CJK, etc.).
    For scripts without case distinctions the word is returned unchanged.
    """
    if not word:
        return word
    return word[0].upper() + word[1:]
