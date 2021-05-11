'''
fanTools : Tools Created by xfan
For coding efficiency
second edition:
2021/04/29
'''
import jieba
from difflib import SequenceMatcher
import pymongo
import re
import time
import random
import json
from operator import itemgetter
import collections

# 因为太过简易使用所以暂时不写做func但又很好用的包：
# 1 - synonyms,

'''part_0 : 杂活王'''


def lineGap_xf():
    """
    将打印结果用线隔开
    """
    print()
    print("--------------------------------------------")
    print()


def time_calc(func):
    """
    decorator(装饰器)的例子 : 计算函数的执行时间
    装饰器功能 : 当每个函数都需要执行一个相同操作，可以把该操作提出来
    装饰器的使用方法 : 在需要使用该装饰器的函数前@一下就可以
    """
    def wrapper(*args, **kargs):
        start_time = time.time()
        f = func(*args, **kargs)
        exec_time = time.time() - start_time
        print("{}'s exec_time:".format(func), exec_time)
        return f
    return wrapper


'''part_1 : 数据文件读写与处理'''


def read_file_to_list(filepath,read_all =True,read_many = 0,shuffle=False):
    """
    func : 将文件读入内存成list
    输入 : 文件路径读入文件
    输出 : 以行为单位（\n结尾）的List
    """
    if read_all:
        with open(filepath, 'r', encoding='utf-8')as f:
            list_all = [x.rstrip('\n') for x in f.readlines()]
        if shuffle:
            random.shuffle(list_all)
        return list_all
    elif read_many:
        tag = 0
        list_all = []
        with open(filepath, 'r', encoding='utf-8')as f:
            for x in f.readlines():
                list_all.append(x.rstrip('\n'))
                tag += 1
                if tag == read_many: break
        if shuffle:
            random.shuffle(list_all)
        return list_all


def write_list_to_file(data_list, filepath, add_breakline=True):
    """
    func : 将内存的list写入file
    输入 : 数据的List，文件路径，标志位 - 是否要换行
    输出 : 文件写入
    """
    with open(filepath, 'a+', encoding='utf-8') as fout:
        if add_breakline:
            fout.writelines([str(x) + '\n' for x in data_list])
        else:
            fout.writelines([str(x) for x in data_list])


def append_list_to_file(data_list, filepath, add_breakline=True):
    """
    func : 将内存的list追加入file
    输入 : 数据的List，文件路径，标志位 - 是否要换行
    输出 : 文件写入
    """
    with open(filepath, 'a+', encoding='utf-8') as fout:
        if add_breakline:
            fout.writelines([str(x) + '\n' for x in data_list])
        else:
            fout.writelines([str(x) for x in data_list])


def read_json_lines_toList(filename):
    """
    func : 将json文件读入内存成list
    输入 : 文件路径读入文件
    输出 : 以行为单位（\n结尾）的List
    """
    data_list = []
    with open(filename) as f:
        for line in f:
            data_list.append(json.loads(line))
    return data_list


def write_json_lines(data, filename):
    """
    func : 将list写入json
    输入 : data_list，文件路径读入文件
    输出 : 写入json
    """
    with open(filename,mode='w', encoding='utf-8',) as f:
        for di in data:
            print(json.dumps(di, ensure_ascii=False), file=f)


def append_json_lines(data, filename):
    """
    func : 将list追加入json
    输入 : data_list，文件路径读入文件
    输出 : 写入json
    """
    with open(filename,mode='a+', encoding='utf-8',) as f:
        for di in data:
            print(json.dumps(di, ensure_ascii=False), file=f)


def write_strLineTo_file(filepath, str):
    """
    func : 将一行字符串写入指定的文件之中
    输入 : 将要写入的文件路径和待写入的一行字符串
    输出 : 无，只是以追加形式将字符串写入文件中，自动换行
    """
    with open(filepath, 'a+', encoding='utf-8')as f:
        f.write(str)
        f.write("\n")


def file_showOneLine_toStr(filepath):
    """
    func : 读取文件第一行并显示并返回
    输入 : 文件路径读入文件
    输出 : 以行为单位（\n结尾）的List
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore')as f:
        str1 = f.readline()  # 若此时继续调用readline，则往下一行接着读
        print(str1)
        return str1


'''part_2 : nlp常用工具'''


def str_countChar_to_dict(string):
    """
    func : 字符串的字符数量统计:然后统计每个字符的出现次数，输出无序的dict
    输入 : 一个字符串str
    输出 : 一个已按键排序的包含所有字符统计结果的dict
    """
    d = dict()
    for ch in string:
        d[ch] = d.get(ch, 0) + 1  # 千万不要使用d[ch] =d[ch] + 1，否则字典没有ch的key，出错
    return d


def str_countChar_TOP10_to_list(string):
    """
    func : 字符串的字符数量统计:然后统计每个字符的出现次数，输出有序的list，取top10
    输入 : 一个字符串str
    输出 : 一个list里面的元素为排序好的字符统计结果的pair，取top10
    """
    frequences = collections.Counter(string)
    print("TOP10 : ", frequences.most_common(10))  # 输出top10
    return frequences.most_common(10)


def sentence_jieba_wordCount_to_list_printTOP5(sentence):
    """
    func : 中文句子的词汇数量统计:然后统计每个字符的出现次数，输出有序的dict，取top10
    输入 : 一个中文字符串句子sentence
    输出 : 一个list里面的元素为排序好的字符统计结果的pair，取top10
    """
    words = jieba.lcut(sentence)
    counts = {}
    for word in words:
        if len(word) == 1:
            continue
        else:
            counts[word] = counts.get(word, 0) + 1
    items = list(counts.items())
    items.sort(key=lambda x: x[1], reverse=True)  # list的排序
    for i in range(5):
        word, count = items[i]
        print("TOP5 :", "{0:<10}{1:>5}".format(word, count))
    return items


def seq_similarity_sequenceMatcher(seq1, seq2):
    """
    func : 引用ratio方法，返回序列相似性的度量
    输入：两个序列 - 字符串/list/tuple
    输出：一个float数字 - 两个序列的相似度
    """
    return SequenceMatcher(None, seq1, seq2)


def sentence_similarity_Jaccrad(model, reference):
    """
    func : 使用Jaccrad计算字符串的相似度
    输入 ：两个字符串 - terms_reference为源句子，terms_model为候选句子
    输出 ：一个float数字 - 两个字符串的相似度
    """
    terms_reference = jieba.cut(reference)  # 默认精准模式
    terms_model = jieba.cut(model)
    grams_reference = set(terms_reference)  # 去重；如果不需要就改为list
    grams_model = set(terms_model)
    temp = 0
    for i in grams_reference:
        if i in grams_model:
            temp = temp + 1
    fenmu = len(grams_model) + len(grams_reference) - temp  # 并集
    jaccard_coefficient = float(temp / fenmu)  # 交集
    return jaccard_coefficient


def getChineseSentence(word):
    """
    func : 得到字符串中的中文
    输入 : str
    输出 : str
    """
    return ''.join(re.findall(u'[\u4e00-\u9fa5]', word))


def KMP(s, p):
    """
    s为主串
    p为模式串
    如果t里有p，返回打头下标
    """
    nex = __getNext(p)
    i = j = 0  # 分别是s和p的指针
    while i < len(s) and j < len(p):
        if j == -1 or s[i] == p[j]:  # j==-1是由于j=next[j]产生
            i += 1
            j += 1
        else:
            j = nex[j]

    if j == len(p):  # 匹配到了
        return i - j
    else:
        return -1


def __getNext(p):
    """
    p为模式串
    返回next数组，即部分匹配表
    等同于从模式字符串的第1位(注意，不包括第0位)开始对自身进行匹配运算。
    """
    nex = [0] * len(p)
    nex[0] = -1
    i = 0
    j = -1
    while i < len(p) - 1:  # len(p)-1防止越界，因为nex前面插入了-1
        if j == -1 or p[i] == p[j]:
            i += 1
            j += 1
            nex[i] = j  # 这是最大的不同：记录next[i]
        else:
            j = nex[j]
    return nex


'''part_3 : 数据库常用操作'''


@time_calc
def mongo_if_get_cndbPedia():
    """
    func : 判断是否连接cndbpedia
    输入 : None
    输出 : db对象，可以用collection = db.triples获取所有三元组
    """
    client = pymongo.MongoClient('mongodb://gdmdbuser:6QEUI8dhnq@10.176.40.106:27017')
    db = client.cndbpedia
    collection = db.triples
    if (collection.find({"s": "木蝴蝶（中药）"}).count()) == 21:
        print("cndbPedia Connected")
    else:
        print("cndePedia connect failed")
    return db


def cndb_givenS_findAllP_toDict(s):
    """
    func : 给出实体将其在cndbPedia中对应的关系
    输入 : str-实体s的字符串
    输出 : dict-包含其在cndbpedia对应所有的关系
    """
    client = pymongo.MongoClient('mongodb://gdmdbuser:6QEUI8dhnq@10.176.40.106:27017')
    db = client.cndbpedia
    collection = db.triples
    results = collection.find({"s": s})
    d = {}
    for result in results:
        d[result["p"]] = d.get(result["p"], 0) + 1
    return d


def cndb_countNumofP_toDict(p):
    """
    func : 给出关系统计其在cndbPedia中对应的数量
    输入 : str-关系p的字符串
    输出 : dict-包含该关系其在cndbpedia对应的数量
    """
    client = pymongo.MongoClient('mongodb://gdmdbuser:6QEUI8dhnq@10.176.40.106:27017')
    db = client.cndbpedia
    collection = db.triples
    results = collection.find({"p": p})
    d = {}
    for result in results:
        d[result["p"]] = d.get(result["p"], 0) + 1
    return d


def cndb_givenP_findTriples_toList(p, find_part, find_all=False):
    """
    func : 给出关系取到将其在cndbPedia中对应的triples
    输入 : str-关系p的字符串,fing_part可以为取部分,find_all为真时取全部
    输出 : list-包含该关系其在cndbpedia对应的triple，find_part为真时取find_part个
    """
    client = pymongo.MongoClient('mongodb://gdmdbuser:6QEUI8dhnq@10.176.40.106:27017')
    db = client.cndbpedia
    collection = db.triples
    results = collection.find({"p": p})
    if find_all:
        list_all1 = []
        for result in results:
            list_part = [result["s"], result["p"], result["o"]]
            list_all1.append(list_part)
        return list_all1
    elif find_part:
        list_all2 = []
        tag = 0
        for result in results:
            tag += 1
            list_part = [result["s"], result["p"], result["o"]]
            list_all2.append(list_part)
            if tag == find_part: break
        return list_all2

# algorithms:
def binary_search(target,num_list):
    """
    func : 二分查找，要求list有序
    输入 : target - 要查找的数字，list-待查找序列
    输出 : 对应的target所在的下标
    """
    low = 0
    high = len(num_list) - 1
    while low<= high:
        mid = (low+high)//2
        item = num_list[mid]
        if target == item:
            return mid
        elif target < item:
            high = mid - 1
        else:
            low = mid +1
    print("not found!")
    return -1

def main():
    pass


if __name__ == '__main__':
    main()
