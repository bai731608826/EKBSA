# coding: utf-8
"""
KnowledgeGraph
"""
import os
import re
import requests
import json
import brain.config as config
import pkuseg
import numpy as np
from ltp import LTP
import copy
from zhon.hanzi import punctuation
import string

def reduce(list):
    temp = []
    for item in list:
        for i in item:
            temp.append(i)
    return temp

class Graph(object):
    """docstring for Graph"""

    def __init__(self, n):
        super(Graph, self).__init__()
        self.n = n
        self.link_list = []
        self.vis = [0] * self.n
        for i in range(self.n):
            self.link_list.append([])

    def add_edge(self, u, v):
        if u == v:
            return
        self.link_list[u].append(v)
        self.link_list[v].append(u)

    def bfs(self, start, dist):
        que = [start]
        self.vis[start] = 1
        for _ in range(dist):
            que2 = []
            for u in que:
                # self.vis[u] = 1
                for v in self.link_list[u]:
                    if self.vis[v]:
                        continue
                    que2.append(v)
                    self.vis[v] = 1
            que = copy.deepcopy(que2)

    def solve(self, start, dist):
        self.vis = [0] * self.n
        self.bfs(start, dist)
        self.vis[0] = 1
        return copy.deepcopy(self.vis)

class KnowledgeGraph(object):
    """
    spo_files - list of Path of *.spo files, or default kg name. e.g., ['HowNet']
    """

    def __init__(self, spo_files, dist, predicate=False):
        self.predicate = predicate
        self.spo_file_paths = [config.KGS.get(f, f) for f in spo_files]
        self.lookup_table = self._create_lookup_table()
        self.segment_vocab = list(self.lookup_table.keys()) + config.NEVER_SPLIT_TAG
        #self.tokenizer = pkuseg.pkuseg(model_name="default", postag=False, user_dict=self.segment_vocab)
        self.special_tags = set(config.NEVER_SPLIT_TAG)
        self.ltp = LTP()
        self.ltp.add_words(self.segment_vocab)
        self.d = dist
        #self.ltp = LTP()

    def _create_lookup_table(self):
        lookup_table = {}
        with open('./brain/kgs/emotion_word.txt', 'r', encoding='utf-8') as f:
            text = f.readlines()
            for item in text:
                item = item.replace('\n', '').split('\t')
                lookup_table[item[0]] = [['情感', item[1]]]

        for spo_path in self.spo_file_paths:
            print("[KnowledgeGraph] Loading spo from {}".format(spo_path))
            with open(spo_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        subj, pred, obje = line.strip().split("\t")
                    except:
                        print("[KnowledgeGraph] Bad spo:", line)
                    if self.predicate:
                        value = [pred, obje]
                    else:
                        value = [obje]
                    if subj in lookup_table.keys():
                        lookup_table[subj].append(value)
                    else:
                        lookup_table[subj] = []
                        lookup_table[subj].append(value)

        return lookup_table


    def add_knowledge_with_vm(self, sent_batch, max_entities=config.MAX_ENTITIES, add_pad=True, max_length=128):
        """
        input: sent_batch - list of sentences, e.g., ["abcd", "efgh"]
        return: know_sent_batch - list of sentences with entites embedding  嵌入实体的句子列表
                position_batch - list of position index of each character. 每个字的位置索引
                visible_matrix_batch - list of visible matrixs  可见矩阵
                seg_batch - list of segment tags
        """
        split_sent_batch = []
        sdp_batch = []
        word_pos_batch = []
        #分词、句法依存分析
        for sent in sent_batch:
            seg_token, hidden = self.ltp.seg([sent])
            sdp = self.ltp.sdp(hidden, mode='graph')
            seg_token[0].insert(0, '[CLS]')
            split_sent_batch.append(seg_token[0])
            sdp_batch.append(sdp[0])

        know_sent_batch = []
        position_batch = []
        visible_matrix_batch = []
        seg_batch = []
        dep_matrix_batch = []

        for split_sent in split_sent_batch:
            #构建句法依存图
            G = Graph(len(split_sent))
            for token in sdp_batch[0]:
                if token[2] == 'Root':
                    continue
                G.add_edge(token[0], token[1])

            # create tree
            sent_tree = []
            dep_tree = []
            pos_idx_tree = []  #soft position
            abs_idx_tree = []  #hard position
            pos_idx = -1
            abs_idx = -1
            abs_idx_src = []
            ntokens = []

            for token in split_sent:
                entities = list(self.lookup_table.get(token, []))[:max_entities]
                sent_tree.append((token, entities))
                length = 0
                if token == '[CLS]':
                    ntokens.append(token)
                    dep_tree.append((1, length))
                else:
                    ntokens.extend(list(token))
                    for en in entities:
                        for e in en:
                            length += len(e)
                            ntokens.extend(list(e))
                    dep_tree.append((len(token), length))

                if token in self.special_tags:
                    token_pos_idx = [pos_idx + 1]  # token的位置信息
                    token_abs_idx = [abs_idx + 1]
                else:
                    token_pos_idx = [pos_idx + i for i in range(1, len(token) + 1)]
                    token_abs_idx = [abs_idx + i for i in range(1, len(token) + 1)]
                abs_idx = token_abs_idx[-1]

                entities_pos_idx = []
                entities_abs_idx = []

                for ent in entities:
                    temp_idx = token_pos_idx[-1]
                    ent_pos_idx = []
                    for e in ent:
                        e_pos_idx = [temp_idx + i for i in range(1, len(e) + 1)]
                        ent_pos_idx.append(e_pos_idx)
                        temp_idx = e_pos_idx[-1]

                        ent_abs_idx = [abs_idx + i for i in range(1, len(e) + 1)]
                        abs_idx = ent_abs_idx[-1]
                        entities_abs_idx.append(ent_abs_idx)
                    #ent_pos_idx = [token_pos_idx[-1] + i for i in range(1, len(ent) + 1)]
                    entities_pos_idx.append(ent_pos_idx)
                    #ent_abs_idx = [abs_idx + i for i in range(1, len(ent) + 1)]
                    #abs_idx = ent_abs_idx[-1]
                    #entities_abs_i dx.append(ent_abs_idx)

                pos_idx_tree.append((token_pos_idx, entities_pos_idx))
                pos_idx = token_pos_idx[-1]
                abs_idx_tree.append((token_abs_idx, entities_abs_idx))
                abs_idx_src += token_abs_idx




            # Get know_sent and pos
            know_sent = []
            pos = []
            seg = []
            for i in range(len(sent_tree)):
                word = sent_tree[i][0]
                pos += pos_idx_tree[i][0]
                if word in self.special_tags:
                    know_sent += [word]
                    seg += [1]
                elif len(sent_tree[i][1]) == 0:
                    add_word = list(word)
                    know_sent += add_word
                    seg += [1] * len(add_word)
                else:
                    add_word = list(word)
                    know_sent += add_word
                    seg += [1] * len(add_word)
                    for j in range(len(pos_idx_tree[i][1])):
                        for k in range(len(pos_idx_tree[i][1][j])):
                                pos.extend(pos_idx_tree[i][1][j][k])
                    for j in range(len(sent_tree[i][1])):
                        for k in range(len(sent_tree[i][1][j])):
                            add_word = list(sent_tree[i][1][j][k])
                            know_sent += add_word
                            seg += [2] * len(add_word)
            token_num = len(know_sent)

            index = []
            i = 0
            for item in dep_tree:
                index.append(i)
                i = i + item[0] + item[1]

            if len(split_sent) > 7:
                dist = 2
            else:
                dist = 1

            dep_matrix = np.zeros((token_num, token_num))
            for i, token in enumerate(split_sent):
                vis = G.solve(i, dist)  # 限定距离内的点
                if i - 1 >= 0:
                    vis_tmp = G.solve(i - 1, dist)
                    for j in range(len(vis_tmp)):
                        vis[j] |= vis_tmp[j]

                if i + 1 < len(split_sent):
                    vis_tmp = G.solve(i + 1, dist)
                    for j in range(len(vis_tmp)):
                        vis[j] |= vis_tmp[j]
                
                for j in range(len(vis)):
                    if vis[j] == 1:

                        for m in range(dep_tree[i][0] + dep_tree[i][1]):
                            for k in range(dep_tree[j][0] + dep_tree[j][1]):
                                dep_matrix[index[i]+m][index[j]+k] = 1
                        """
                        for m in range(dep_tree[i][0]):
                            for k in range(dep_tree[j][0]):
                                dep_matrix[index[i]+m][index[j]+k] = 1
                        """




            quote_idx = []
            for i, w in enumerate(ntokens):
            # if w in '!\"#$%&\'()*+,.-/:;<=>?@[\\]^_。':
                if w in string.punctuation or w in punctuation or w == '[CLS]':
                    quote_idx.append(i)

            for i in range(len(ntokens)):
                for j in quote_idx:
                    dep_matrix[i][j] = 1
            # Calculate visible matrix
            visible_matrix = np.zeros((token_num, token_num))
            for item in abs_idx_tree:
                src_ids = item[0]
                for id in src_ids:
                    visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                    visible_matrix[id, visible_abs_idx] = 1
                for ent in item[1]:
                    for id in ent:
                        #visible_abs_idx = ent + src_ids
                        e = reduce(item[1])
                        visible_abs_idx = e + src_ids
                        visible_matrix[id, visible_abs_idx] = 1
            for i in range(np.size(dep_matrix, 1)):
                dep_matrix[0][i] = 1
            src_length = len(know_sent)
            if len(know_sent) < max_length:
                pad_num = max_length - src_length
                know_sent += [config.PAD_TOKEN] * pad_num
                seg += [0] * pad_num
                pos += [max_length - 1] * pad_num
                visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
                dep_matrix = np.pad(dep_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
            else:
                know_sent = know_sent[:max_length]
                seg = seg[:max_length]
                pos = pos[:max_length]
                visible_matrix = visible_matrix[:max_length, :max_length]
                dep_matrix = dep_matrix[:max_length, :max_length]

            know_sent_batch.append(know_sent)
            position_batch.append(pos)
            visible_matrix_batch.append(visible_matrix)
            dep_matrix_batch.append(dep_matrix)
            seg_batch.append(seg)

        return know_sent_batch, position_batch, visible_matrix_batch, seg_batch, dep_matrix_batch

