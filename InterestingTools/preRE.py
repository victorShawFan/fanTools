import jiagu

from fanTools import *


def re_use_jiagu(text):
    '''
    input : str
    output: list
    '''
    knowledge = jiagu.knowledge(text)
    print(knowledge)
    return knowledge


def main():
    '''
    通用百科语料训练的RE，定义关系如 作者 导演 毕业院校 等，对通用语料可以做一个preRE，以填充数据
    '''
    fin_path = './testREsenteces.txt'
    fout_path = './testREresults.json'
    sentence_list = read_file_to_list(fin_path)
    for i in sentence_list:
        triples = re_use_jiagu(i)
        dict_tri = {}
        dict_tri['text'] = i
        dict_tri['spo_list'] = []
        for j in triples:
            single_dict = {}
            single_dict['label'] = j[1]
            single_dict['s'] = j[0]
            single_dict['o'] = j[2]
            dict_tri['spo_list'].append(single_dict)
        with open(fout_path, 'a+', encoding='utf-8')as fout:
            fout.write("{}\n".format(json.dumps(dict_tri, ensure_ascii=False)))


if __name__ == '__main__':
    main()
