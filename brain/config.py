import os

FILE_DIR_PATH = os.path.dirname(os.path.abspath(__file__))

KGS = {
    'HowNet': os.path.join(FILE_DIR_PATH, 'kgs/HowNet.txt'),
    'CnDbpedia': os.path.join(FILE_DIR_PATH, 'kgs/CnDbpedia.txt'),
}

MAX_ENTITIES = 2

# Special token words.
PAD_TOKEN = '[PAD]'#用来补全句子
UNK_TOKEN = '[UNK]'
CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'
MASK_TOKEN = '[MASK]'
SUB_TOKEN = '[SUB]'
PRE_TOKEN = '[PRE]'
OBJ_TOKEN = '[OBJ]'
ENT_TOKEN = '[ENT]'
REL_TOKEN = '[REL]'
WORD_TOKEN = '[WORD]'

NEVER_SPLIT_TAG = [
    PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN,
    ENT_TOKEN, SUB_TOKEN, PRE_TOKEN, OBJ_TOKEN, ENT_TOKEN,
    REL_TOKEN, WORD_TOKEN
]

WORD_POS = ['a', 'idiom', 'n', 'nh', 'ni', 'nz', 'v', 'z']
