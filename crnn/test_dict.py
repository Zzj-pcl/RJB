# ANSI文件转UTF-8
import codecs
import os

file_path = "H:\hfw\lab/alphabet.txt"
# 功能：测试字典表读取是否正确
def read_alphabet(filename):
    file = []
    with open(filename, 'r', encoding='utf-8') as f:
        while True:
            raw = f.readline()
            if not raw:
                break
            file.append(raw)
    idx2symbol = [s.strip('\n') for s in file]
    for i in range(92):
        idx2symbol[i] = idx2symbol[i][1:]
    idx2symbol = raw.split()
    idx2symbol.insert(0, '<pad>')
    idx2symbol.insert(1, '<GO>')
    idx2symbol.insert(2, '<EOS>')
    idx2symbol.append(' ')
    symbol2idx = {}
    for idx, symbol in enumerate(idx2symbol):
        symbol2idx[symbol] = idx
    return idx2symbol, symbol2idx

idx2symbol, symbol2idx = read_alphabet(file_path)
print(idx2symbol)
print(len(idx2symbol))