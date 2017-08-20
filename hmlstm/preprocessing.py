import re
import numpy as np
from string import ascii_lowercase


def text(text_path, truncate_len, step_size, batch_size, num_chars=None):
    signals = load_text(text_path, truncate_len, step_size, batch_size, num_chars)

    hot = [(one_hot_encode(intext), one_hot_encode(outtext))
           for intext, outtext in signals]

    return hot


def load_text(text_path, truncate_len, step_size, batch_size, num_chars):
    with open(text_path, 'r') as f:
        text = f.read(num_chars)
        text = text.replace('\n', ' ')
        text = re.sub(' +', ' ', text).lower()  # removes +?

    signals = []
    start = 0
    while start + truncate_len < len(text):
        intext = text[start:start + truncate_len]
        outtext = text[start + 1:start + truncate_len + 1]
        signals.append((intext, outtext))
        start += step_size

    return signals


def one_hot_encode(text):
    out = np.zeros((len(text), 27))

    def get_index(char):
        try:
            return ascii_lowercase.index(char)
        except:
            return 26

    for i, t in enumerate(text):
        out[i, get_index(t)] = 1
    # out = text
    return out


def get_text(encoding):
    prediction = ''

    for char in np.squeeze(encoding):
        max_likelihood = np.where(char == np.max(char))[0][0]
        if max_likelihood < 26:
            prediction += ascii_lowercase[max_likelihood]
        elif max_likelihood == 26:
            prediction += ' '

    return prediction


def prepare_inputs(batch_size=10,
                   truncate_len=1000,
                   text_path='text8.txt',
                   step_size=None,
                   num_batches=None):
    """
    :param batch_size: # of text segments of length truncate_len per batch
    :param truncate_len: length of input sequence for RNN (how many input neurons involved)
    :param text_path: path where to load the text from
    :param step_size: # of chars to advance for the next segment
    :param num_batches: if not given, then ceil(num_segments / batch_size), if given, only batch_size * num_batches * truncate_len of he text is loaded
    :return: batches_in: contains the input data, each batch contains truncate_len chars, with a relative shift of step_size

    First the a number of segments of length truncate_len and stride step_size are created.
    There should be roughly (text_length - truncate_len)/step_size of them
    These segments are then collected into batches of batch_size

    A batch is a collection of training samples that are used together to update the network weights.
    If the batch contains ALL training samples, then we have batch grad descent (expensive and doesn't
    generalize well)
    Usually batches are of small size ~10, this is mini-batch GD. If we use one sample at a time
    (batch_size=1) this is what ppl usually call stochastic GD (SGD).

    So our segmentation is: we generate n_samples training examples of length truncate_len.
    We then group these into n_batches batches of size batch_size
    (so ideally, num_batches * batch_size = n_samples, assuming divisibility)
    """
    if step_size is None:
        step_size = truncate_len // 2  # seems like a pretty random default...

    if num_batches is None:
        y = text(text_path, truncate_len, step_size, batch_size)  # here batch_size does not do anything
        num_batches = len(y) // batch_size
    elif num_batches is not None:
        if step_size > truncate_len:
            raise ValueError('Step size cannot be greater than truncate length')
        num_chars = batch_size * num_batches * truncate_len
        y = text(text_path, truncate_len, step_size, batch_size, num_chars)

    batches_in = []
    batches_out = []

    for batch_number in range(num_batches):
        start = batch_number * batch_size
        end = start + batch_size
        batches_in.append([i for i, _ in y[start:end]])
        batches_out.append([o for _, o in y[start:end]])

    return batches_in, batches_out


def convert_to_batches(signals, batch_size=10, steps_ahead=1):
    start = 0
    batches_in = []
    batches_out = []
    while start + batch_size < len(signals):
        batch = signals[start: start + batch_size]

        batches_in.append(np.array([s[:-steps_ahead] for s in batch]).reshape(batch_size, -1, 1))
        batches_out.append(np.array([s[steps_ahead:] for s in batch]).reshape(batch_size, -1, 1))

        start += batch_size

    return np.array(batches_in), np.array(batches_out)
