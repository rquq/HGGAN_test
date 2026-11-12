"""Literal corpus CER/WER and mean normalized edit distance (no CTC collapse)."""


def _distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j-1] + 1, previous[j-1] + (x != y)))
        previous = current
    return previous[-1]


class RecognitionMetrics:
    def __init__(self):
        self.samples = self.char_edits = self.target_chars = 0
        self.word_edits = self.target_words = self.exact_matches = 0
        self.ned_sum = 0.

    def update(self, target, prediction):
        if not isinstance(target, str) or not isinstance(prediction, str):
            raise TypeError('target and prediction must be literal strings')
        if not target:
            raise ValueError('target must be a nonempty literal transcript')
        edits = _distance(target, prediction)
        self.samples += 1
        self.char_edits += edits
        self.target_chars += len(target)
        self.word_edits += _distance(target.split(), prediction.split())
        self.target_words += len(target.split())
        self.ned_sum += edits / max(len(target), len(prediction), 1)
        self.exact_matches += target == prediction

    def compute(self):
        if not self.target_words:
            raise ValueError('corpus metrics require at least one target word')
        cer = self.char_edits / max(self.target_chars, 1)
        wer = self.word_edits / max(self.target_words, 1)
        ned = self.ned_sum / max(self.samples, 1)
        # IAM items are word crops: accuracy is exact literal transcript match,
        # not 1-WER (which can be negative when predictions insert many words).
        accuracy = self.exact_matches / self.samples
        return dict(samples=self.samples, char_edits=self.char_edits, target_chars=self.target_chars,
                    word_edits=self.word_edits, target_words=self.target_words,
                    exact_matches=self.exact_matches, ned_sum=self.ned_sum,
                    cer=cer, wer=wer, ned=ned, word_accuracy=accuracy,
                    cer_percent=100*cer, wer_percent=100*wer, ned_percent=100*ned,
                    word_accuracy_percent=100*accuracy)
