#coding:utf-8
import time
import os
import sys
import codecs
import chardet
import re
from zhon.hanzi import punctuation
import string

def getFileInfo(filename):
    path, name = os.path.split(filename)
    (name, type) = os.path.splitext(name)
    return path, name, type

def getFileDetailInfo(filename, speaker):
    # /mnt/nlpr/DATA/Audio/Chinese/863_2/863ChongQing/CQF021/CQF021s0085a.wav
    path, name = os.path.split(filename)
    name, type = os.path.splitext(name)
    splits = filename.split('/')
    return path, splits[speaker], name, type

def get_grid(filename):
    """Read the file and return the content."""
    with codecs.open(filename, 'r', 'utf-8') as f:
        c = f.readlines()
        c = c.replace('\r', '')
    return ''.join(c)

def detectEncoding(f):
    """
    This helper method returns the file encoding corresponding to path f.
    This handles UTF-8, which is itself an ASCII extension, so also ASCII.
    """
    rf = open(f, 'rb')
    data = rf.read()
    return chardet.detect(data)['encoding']

def cleantxt(text):
    punc = punctuation + string.punctuation
    trantab1 = str.maketrans({key: None for key in punc})
    text = text.translate(trantab1)
    return text

def normalize_english_text(text):
    text = text.lower()
    remove = str.maketrans('', '', string.punctuation) 
    text = text.translate(remove)
    return text

def to_str(bytes_or_str, encoding = 'utf-8'):
  if isinstance(bytes_or_str, bytes):
      value = bytes_or_str.decode(encoding, 'ignore')
  else:
      value = bytes_or_str
  return value # Instance of str

def to_bytes(bytes_or_str, encoding = 'utf-8'):
  if isinstance(bytes_or_str, str):
      print(encoding)
      value = bytes_or_str.encode(encoding, 'ignore')
  else:
      value = bytes_or_str
  return value # Instance of bytes


def FileSize(audio_path):
    return os.path.getsize(audio_path)