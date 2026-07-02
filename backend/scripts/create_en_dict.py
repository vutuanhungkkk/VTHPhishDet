"""Create en_dict.txt for RapidOCR English recognition model."""
import os

chars = list('0123456789abcdefghijklmnopqrstuvwxyz')
chars += list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
chars += list('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
chars += [' ']

dict_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ocr', 'en_dict.txt')
dict_path = os.path.normpath(dict_path)

with open(dict_path, 'w', encoding='utf-8') as f:
    for c in chars:
        f.write(c + '\n')

print(f"Wrote {len(chars)} characters to {dict_path}")
