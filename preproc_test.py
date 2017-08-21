# from hmlstm import HMLSTMNetwork, prepare_inputs, get_text, load_text
from hmlstm.preprocessing import load_text
import re
from pprint import pprint

batch_size = 10
num_batches = 10
truncate_len = 30
step_size = 5
text_path = 'sample.txt'
num_chars = None

with open(text_path, 'r') as f:
    text = f.read(num_chars)
    text = text.replace('\n', ' ')
    text = re.sub(' +', ' ', text).lower()

# print(len(text))
pprint(text)
y = load_text(text_path, truncate_len, step_size, batch_size, num_chars)

for n, sig in enumerate(y):
    print(n)
    pprint(sig)

