text_set = []
set = []
path = r'D:\PycharmProjects\DM_project\K-BERT\KBert_emotion\brain\kgs\emotion_word.txt'

with open(path, 'r', encoding='utf-8') as f:
    text = f.readlines()
    for item in text:
        item = item.replace('\n', '').replace(' ','').split('\t')
        if item[1] == '积极':
            text_set.append([item[0], '正向'])
        elif item[1] == '消极':
            text_set.append([item[0], '负向'])
        else:
            text_set.append(item)
path = r'D:\PycharmProjects\DM_project\K-BERT\KBert_emotion\brain\kgs\emotion_word_new.txt'
with open(path, 'w', encoding='utf-8') as f:
    for item in text_set:
        f.write(item[0]+'\t'+item[1])
        f.write('\n')
