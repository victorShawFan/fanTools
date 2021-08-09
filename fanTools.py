'''
fanTools : Tools Created by xfan
For coding efficiency
sixth edition:
2021/07/29
'''
import collections
import datetime
import gc
import json
import os
import random
import re
import string
import sys
import time
from difflib import SequenceMatcher
from string import punctuation

import jieba
import jsonlines
import numpy as np
import pymongo
import tensorflow as tf

# 因为太过简易使用所以暂时不写做func但又很好用的包：
# 1 - synonyms,
import torch

'''part_0 : 杂活王'''

'''part_0-1 : 提性能(省内存和加速)'''


def how_big_is(x):
    '''显示一个变量占多少字节'''
    print(f'{x} is ', sys.getsizeof(x), 'Bytes')
    return sys.getsizeof(x)


def how2watchCodeMemory():
    '''如何使用memory-profiler查看进程或者代码块的内存消耗'''
    print('''
    1. 装饰器+第三方：
    使用装饰器来分析制定的函数，然后使用特殊脚本运行该脚本（在这种情况下，使用特定参数到Python 解释器）
    写代码时在要查看的函数钱前@了之后，通过shell调用程序，加载memory_profiler模块并打印内存使用日志。
    python -m memory_profiler example.py
    2. api调用：
    memory_profiler公开了许多要在第三方代码中使用的函数：memory_usage返回一段时间内的内存使用情况。
    memory_usage(proc=-1, interval=.1, timeout=None)
    proc代表监控的内容，可以是进程号；
    interval代表间隔0.1秒获取内存消耗；
    通过API调用，可以直接在代码内部统计内存的使用。
    def f(a, n=100): ...
    from memory_profiler import memory_usage 
    memory_usage((f, (1,), {'n' : int(1e6)}))
    ''')


def reduce_memory(df):
    '''
    func: 内存压缩 - 将一个pandas的Dataframe按每一列压缩至占空间最小的数据类型
    输入 : 一个Dataframe
    输出 : 一个Dataframe
    '''
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print("Dataframe reducetion:")
    print('-' * 70)
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                    print(f'No.{col} column has converted to np.int8')
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                    print(f'No.{col} column has converted to np.int16')
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                    print(f'No.{col} column has converted to np.int32')
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
                    print(f'No.{col} column has converted to np.int64')
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                    print(f'No.{col} column has converted to np.float16')
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                    print(f'No.{col} column has converted to np.float32')
                else:
                    df[col] = df[col].astype(np.float64)
                    print(f'No.{col} column has converted to np.float64')
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('start_mem : {:.6f} Mb, end_mem : {:.6f} Mb (reduced :{:.6f} %)'.format(start_mem, end_mem,
                                                                                  100 * (
                                                                                          start_mem - end_mem) / start_mem))
    gc.collect()  # 垃圾回收机制，就是把返回处理这些循环引用一共释放掉的对象个数，对已经销毁的对象，Python不会自动释放其占据的内存空间。为了能够充分地利用分配的内存，避免程序跑到一半停止，要时不时地进行内存回收
    print("-" * 70)
    return df


'''part0-2 : 真杂活'''


def del_all_file(path):
    '''
    删除path目录下的所有内容
    '''
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_all_file(c_path)
        else:
            print("del: ", c_path)
            os.remove(c_path)


def when_plot_chinese():
    '''
    plt画图时无法显示中文怎么办
    '''
    print('''
    just add :
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    ''')


def show_me_shortcut(pycharm=True, vscode=False):
    """展示IDE的快捷键
    Args:
        pycharm (bool, optional): 默认使用pycharm. Defaults to True.
        vscode (bool, optional): 默认不使用VSCode. Defaults to False.
    """
    if pycharm:
        print('''
    git查看代码变动 : git diff
    # 代码整理
    快速建立遍历结构 : iter + enter
    一键给代码加入indent : ctrl+Alt+L
    一键整理import : ctrl+Alt+o
    # 代码快捷键
    当前行复制并且新增一行 : ctrl+D
    剪切当前行 : ctrl+X
    替换 : ctrl+R  (全局加shift)
    插入模板 : ctrl+J
    补全代码 : ctrl+shift+enter
    快速修正 : Alt+enter
    跳到代码块开始处 : ctrl+[
    跳到代码块结束处 : ctrl+]
    展开所有代码块 : ctrl+shift+(+)
    收缩所有代码块 : ctrl+shift+(-)
    上下移动选中代码 : ctrl+shift+ ↑ (↓)
    进入列编辑模式 : alt+鼠标
    选中单词 : ctrl+w
    跨行选取相同的单词 : Alt+J
    跳到下一个函数 : Alt + ↓
    # 文件级别处理
    将当前py文件重命名 : shift + F6
    关闭当前py文件 : ctrl + F4
    切换代码窗口查看 : ctrl+tab
    查找文件名 : ctrl+shift+N
    # 查看
    快速查看函数说明文档 : ctrl + q
    快速查看方法实现的内容 : ctrl+shift+i
    查看函数定义源码 —>光标放在函数名 : ctrl+B 
    查看函数参数提示 -> 括号内 : ctrl+P
    查看方法在哪里被调用 : ctrl+Alt+H
    # 命令行
    快速打开万能栏 : 双击shift
    万能命令行 : ctrl+shift+A
    打开快捷键说明文档 : ctrl+shift+A 键入-> keymap
    ''')
    if vscode:
        print('''
              自动indent代码:alt+shift+F
              Ctrl+alt+shift+↑/↓ : 选中多行多光标同时操作
              Alt+↑/↓ : 拖着这一行上下移动
              Alt+shift+↑/↓ : 复制一行
              ''')


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


def print_time_bar():
    '''
    func : 打印当前时间
    '''
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "==========" * 8 + "%s" % nowtime)
    return nowtime


'''part_1 : 数据文件读写与处理'''


# 家卿格式json数据的读写
# with open('dev.json', 'r', encoding='utf-8')as f:
#     lines = f.readlines()
#     for line in lines:
#         data = json.loads(line)  # 此时data为一个dict{text,spo_list:[]}
#         with open("train.json", 'a+',encoding='utf-8')as fout:
#             fout.write(("{}\n".format(json.dumps(data, ensure_ascii=False))))


def read_dictJson2dictList(file):
    '''
    func : 对于数据格式为dictline的json,读取其数据到一个list中
    输入 : 每一行是一个dict的json文件路径
    输出 : 一个全是dict的list
    '''
    huge_dict = []
    dict_num = 0
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            huge_dict.append(data)
            dict_num += 1
    print("number of dicts : ", dict_num)
    return huge_dict


def write2json(data, file_path):
    '''write a dict to jsonfile'''
    with open(file_path, "a+", encoding="utf-8") as f_out:
        f_out.write("{}\n".format(json.dumps(data, ensure_ascii=False)))
    return


def read_bigText(file):
    '''当文件特别大时，将文件做成一个生成器'''
    for row in open(file, "r", encoding="utf-8"):
        yield row


def read_file_to_list(filepath, read_all=True, read_many=0, shuffle=False):
    """
    func : 将文件读入内存成list
    输入 : 文件路径读入文件
    输出 : 以行为单位（\n结尾）的List
    """
    if read_all:
        with open(filepath, 'r', encoding='utf-8') as f:
            list_all = [x.rstrip('\n') for x in f.readlines()]
        if shuffle:
            random.shuffle(list_all)
        return list_all
    elif read_many:
        tag = 0
        list_all = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for x in f.readlines():
                list_all.append(x.rstrip('\n'))
                tag += 1
                if tag == read_many:
                    break
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
    with open(
            filename,
            mode='w',
            encoding='utf-8',
    ) as f:
        for di in data:
            print(json.dumps(di, ensure_ascii=False), file=f)


def append_json_lines(data, filename):
    """
    func : 将list追加入json
    输入 : data_list，文件路径读入文件
    输出 : 写入json
    """
    with open(
            filename,
            mode='a+',
            encoding='utf-8',
    ) as f:
        for di in data:
            print(json.dumps(di, ensure_ascii=False), file=f)


def write_strLineTo_file(filepath, str):
    """
    func : 将一行字符串写入指定的文件之中
    输入 : 将要写入的文件路径和待写入的一行字符串
    输出 : 无，只是以追加形式将字符串写入文件中，自动换行
    """
    with open(filepath, 'a+', encoding='utf-8') as f:
        f.write(str)
        f.write("\n")
    return


def file_showOneLine_toStr(filepath):
    """
    func : 读取文件第一行并显示并返回
    输入 : 文件路径读入文件
    输出 : 以行为单位（\n结尾）的List
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        str1 = f.readline()  # 若此时继续调用readline，则往下一行接着读
        print(str1)
        return str1


'''part_2 : nlp常用工具'''


def foolTokenizer(x):
    '''按空格分词'''
    return re.sub('[%s]' % string.punctuation, "", x).split(" ")


def clean_en_text(text):
    '''英文文本清洗'''
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text


def filterLowFreqWords(arr, MAX_WORDS=10000):
    '''过滤低频词，MAX_WORDS为保留的最高频的词数量'''
    arr = [[x if x < MAX_WORDS else 0 for x in example] for example in arr]
    return arr


def NormalizeText(text):
    '''中文文本清洗'''
    return re.sub('[\u3000\uac00-\ud7ff]+', ' ', text)


def remove_punctuation(str):
    '''去除字符串里所有的标点符号'''
    punc = punctuation + u'.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：'
    str1 = re.sub(r"[{}]".format(punc), '', str)
    return str1


def cleanText(text):
    '''将字符串去除空格和\t和\n等杂质,正则版'''
    text = text.strip()
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\t", "", text)
    text = re.sub(r" ", "", text)
    return text


def cleanStr(str):
    '''将字符串去除空格和\t和\n等杂质，python字符串版'''
    return str.replace('\n', '').replace('\t', '').replace(' ', '')


def remove_herf(str):
    '''去除字符串中的herf'''
    return re.sub('<a[^>]*>', '', str).replace('</a>', '')


def seq_padding(X, maxlen=0, padding=0, self_adapt=False):
    '''对给入的向量做maxlen长度的padding'''
    if self_adapt:
        L = [len(x) for x in X]
        ML = max(L)
        return np.array([
            np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
        ])
    else:
        return np.array([np.concatenate([x, [padding] * (maxlen - len(x))]) if len(x) < maxlen else x for x in X])


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


def seq_similarity_gestalt(seq1, seq2):
    """
    func : 引用gestalt方法，返回序列相似性的度量，其实就和jaccard差不多
    输入：两个序列 - 字符串/list/tuple
    输出：一个float数字 - 两个序列的相似度
    """
    return SequenceMatcher(None, seq1, seq2).ratio()


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


def intToRoman(num):
    '''
    整数转成罗马数字，整理成硬编码表，利用模运算和除法运算
    输入 : 整数
    输出 : 罗马数字
    '''
    THOUSANDS = ["", "M", "MM", "MMM"]
    HUNDREDS = ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"]
    TENS = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"]
    ONES = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]
    return THOUSANDS[num // 1000] + HUNDREDS[num % 1000 //
                                             100] + TENS[num % 100 //
                                                         10] + ONES[num % 10]


def romanToInt(s):
    '''
    罗马数字转成数字
    输入 : 字符串 s - 含有古罗马数字
    输出 : 整数
    '''
    # 构建一个字典记录所有罗马数字子串，注意长度为2的子串记录的值是（实际值 - 子串内左边罗马数字代表的数值）
    d = {
        'I': 1,
        'IV': 3,
        'V': 5,
        'IX': 8,
        'X': 10,
        'XL': 30,
        'L': 50,
        'XC': 80,
        'C': 100,
        'CD': 300,
        'D': 500,
        'CM': 800,
        'M': 1000
    }
    # 遍历整个 ss 的时候判断当前位置和前一个位置的两个字符组成的字符串是否在字典内，如果在就记录值，不在就说明当前位置不存在小数字在前面的情况，直接记录当前位置字符对应值
    return sum(d.get(s[max(i - 1, 0):i + 1], d[n]) for i, n in enumerate(s))
    # 遍历经过 IVIV 的时候先记录 II 的对应值 11 再往前移动一步记录 IVIV 的值 33，加起来正好是 IVIV 的真实值 44。max 函数在这里是为了防止遍历第一个字符的时候出现 [-1:0][−1:0] 的情况


def getChineseSentence(word):
    """
    func : 得到字符串中的中文
    输入 : str
    输出 : str
    """
    return ''.join(re.findall(u'[\u4e00-\u9fa5]', word))


def vector_visualization():
    print('''
    # 先聚类
    y_pred = DBSCAN(eps=eps).fit_predict(embeds)
    # 然后降维
    dim = PCA(n_components=2).fit_transform(embeds)[:, :2]
    # 获得类别总数
    classes = set([i for i in y_pred])

    # 获得聚类信息
    cluster = {}
    for i, j in zip(y_pred, values):
        if i != -1:
            li = cluster.get(i, [])
            li.append(j)
            cluster[i] = li

    # 画图以及保存
    plt.clf()
    plt.figure(figsize=(9, 9))
    plt.title('%s密度聚类效果' % file.strip('.txt'))
    plt.scatter(dim[:, 0], dim[:, 1], c=y_pred)
    for w, (x, y) in zip(values, dim):
        plt.text(x + 0.005, y + 0.005, w)
    plt.savefig(os.path.join(output, '%s.png' % file.strip('.txt')))
    ''')


'''part_3 : 数据库常用操作'''


@time_calc
def mongo_if_get_cndbPedia():
    """
    func : 判断是否连接cndbpedia
    输入 : None
    输出 : db对象，可以用collection = db.triples获取所有三元组
    """
    client = pymongo.MongoClient(
        'mongodb://gdmdbuser:6QEUI8dhnq@10.176.40.106:27017')
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
    client = pymongo.MongoClient(
        'mongodb://gdmdbuser:6QEUI8dhnq@10.176.40.106:27017')
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
    client = pymongo.MongoClient(
        'mongodb://gdmdbuser:6QEUI8dhnq@10.176.40.106:27017')
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
    client = pymongo.MongoClient(
        'mongodb://gdmdbuser:6QEUI8dhnq@10.176.40.106:27017')
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
            if tag == find_part:
                break
        return list_all2


def mongo_subMatch2list(collection, key, value):
    '''
    func : 对于指定的mongodb的colletion，给于key和value，返回所有匹配的结果
    输入 : collection为表名, key为键名,value为要匹配的子串
    输出 : 一个list包含所有该子串匹配的结果
    '''
    results = collection.find({key: {"$regex": value}})
    return list(set([x[key] for x in results]))


'''part_4 : Algorithms'''


def binary_search(target, num_list):
    """
    func : 二分查找，要求list有序
    输入 : target - 要查找的数字，list-待查找序列
    输出 : 对应的target所在的下标
    """
    low = 0
    high = len(num_list) - 1
    while low <= high:
        mid = (low + high) // 2
        item = num_list[mid]
        if target == item:
            return mid
        elif target < item:
            high = mid - 1
        else:
            low = mid + 1
    print("not found!")
    return -1


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


'''part_5 : Parallel Accelerating'''


def parallel_task_run():
    '''多线程与多进程'''
    a = '''
    from multiprocessing import Process,Pool,cpu_count
    [p = Process(target=run_proc, args=('test',))
    p.start()
    p.join() ] # 创建子进程时，只需要传入一个执行函数和函数的参数，创建一个Process实例，用start()方法启动，这样创建进程比fork()还要简单。
join()方法可以等待子进程结束后再继续往下运行，通常用于进程间的同步。
    [p = Pool(cpu_count) # pool默认大小在电脑上是8，最多同时执行8个进程，pool的默认大小为CPU的核数
    for i in range(cpu_count+1): # task 0-7是立刻执行的，而task8要等待前面某个task完成后才执行，这是因为Pool的默认大小在我的电脑上是8，因此，最多同时执行8个进程。这是Pool有意设计的限制，并不是操作系统的限制
        p.apply_async(long_time_task, args=(i,))
    p.close() # 调用join之前必须先调用close，close后就不能继续添加新的process
    p.join() # 对pool对象调用join方法会等待所有子进程执行完毕
    ]
    from threading import Thread # Thead和process用起来差不多 thread = Thread(target=count, args=(1,1))然后start和join
    但是由于，多进程中，同一个变量，各自有一份拷贝存在于每个进程中，互不影响，而多线程中，所有变量都由所有线程共享，
    所以，任何一个变量都可以被任何一个线程修改，因此，线程之间共享数据最大的危险在于多个线程同时改一个变量，把内容给改乱了
    所以用多线程要加锁 lock.acquire() - func() - lock.release()
    多核时候：
    for i in range(multiprocessing.cpu_count()):
    t = threading.Thread(target=loop)
    t.start()
    '''
    print(a)
    pass


import psutil
import platform


def show_computer_info():
    def get_windows_cpu_speed():
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        speed, type = winreg.QueryValueEx(key, "~MHz")
        speed = round(float(speed) / 1024, 1)
        return "{speed} GHz".format(speed=speed)

    def get_cpu_speed():
        osname = platform.system()  # 获取操作系统的名称
        speed = ''
        if osname in ["Windows", "Win32"]:
            speed = get_windows_cpu_speed()
        return speed

    def get_cpu_info():
        cpu1 = psutil.cpu_count()
        cpu2 = psutil.cpu_count(logical=False)
        print("cpu逻辑个数:", cpu1)
        print("cpu真实内核个数:", cpu2)

    def get_mem_info():
        mem = psutil.virtual_memory()
        mem1 = str(mem.total / 1024 / 1024 / 1024)
        mem2 = str(mem.free / 1024 / 1024 / 1024)
        print("内存总数为:", mem1[0:3], "G")
        print("空闲内存总数:", mem2[0:3], "G")

    print("cpu的频率 :", get_cpu_speed())
    get_cpu_info()
    get_mem_info()


'''part_6 : 正则常用函数'''


def usual_regex_Desc(read=True, cheatSheet=False):
    '''常用正则表达书的说明'''
    if read:
        print('''
            Python通过re模块支持正则表达式，re 模块使 Python 语言拥有全部的正则表达式功能
            常用的方法:
            re.match(pattern, string, flags=0)尝试从字符串的起始位置匹配一个模式，如果不是起始位置匹配成功的话，match()就返回none
            re.search(pattern, string, flags=0)扫描整个字符串并返回第一个成功的匹配，匹配成功re.search方法返回一个匹配的对象，否则返回None
            re.match 只匹配字符串的开始，如果字符串开始不符合正则表达式，则匹配失败，函数返回 None，而 re.search 匹配整个字符串，直到找到一个匹配
            re.sub(pattern, repl, string, count=0, flags=0)提供了re.sub用于替换字符串中的匹配项
            re.compile(pattern[, flags])用于编译正则表达式，生成一个正则表达式（ Pattern ）对象;
            re.findall(pattern, string, flags=0)用于在字符串中找到正则表达式所匹配的所有子串，并返回一个列表，如果没有找到匹配的，则返回空列表。
            re.finditer(pattern, string, flags=0)和 findall 类似，在字符串中找到正则表达式所匹配的所有子串，并把它们作为一个迭代器返回
            re.split(pattern, string[, maxsplit=0, flags=0])按照能够匹配的子串将字符串分割后返回列表
            ''')
    if cheatSheet:
        print('''. 
            11. 数字 : 
            验证数字：^[0-9]*$
            验证n位的数字：^\d{n}$
            验证至少n位数字：^\d{n,}$
            验证m-n位的数字：^\d{m,n}$
            验证零和非零开头的数字：^(0|[1-9][0-9]*)$
            验证有两位小数的正实数：^[0-9]+(.[0-9]{2})?$
            验证有1-3位小数的正实数：^[0-9]+(.[0-9]{1,3})?$
            验证非零的正整数：^\+?[1-9][0-9]*$
            验证非零的负整数：^\-[1-9][0-9]*$
            验证非负整数（正整数 + 0） ^\d+$
            验证非正整数（负整数 + 0） ^((-\d+)|(0+))$
            整数：^-?\d+$
            非负浮点数（正浮点数 + 0）：^\d+(\.\d+)?$
            正浮点数 ^(([0-9]+\.[0-9]*[1-9][0-9]*)|([0-9]*[1-9][0-9]*\.[0-9]+)|([0-9]*[1-9][0-9]*))$
            非正浮点数（负浮点数 + 0） ^((-\d+(\.\d+)?)|(0+(\.0+)?))$
            负浮点数 ^(-(([0-9]+\.[0-9]*[1-9][0-9]*)|([0-9]*[1-9][0-9]*\.[0-9]+)|([0-9]*[1-9][0-9]*)))$
            浮点数 ^(-?\d+)(\.\d+)?$
            12. 字符串 : 
            英文和数字：^[A-Za-z0-9]+$ 或 ^[A-Za-z0-9]{4,40}$
            长度为3-20的所有字符：^.{3,20}$
            由26个英文字母组成的字符串：^[A-Za-z]+$
            由26个大写英文字母组成的字符串：^[A-Z]+$
            由26个小写英文字母组成的字符串：^[a-z]+$
            由数字和26个英文字母组成的字符串：^[A-Za-z0-9]+$
            由数字、26个英文字母或者下划线组成的字符串：^\w+$ 或 ^\w{3,20}$
            中文、英文、数字包括下划线：^[\u4E00-\u9FA5A-Za-z0-9_]+$
            中文、英文、数字但不包括下划线等符号：^[\u4E00-\u9FA5A-Za-z0-9]+$ 或 ^[\u4E00-\u9FA5A-Za-z0-9]{2,20}$
            可以输入含有^%&',;=?$\”等字符：[^%&',;=?$\x22]+
            禁止输入含有\~的字符：[^~\x22]+
            13.特殊 :
            URL:/^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$/
            IP 地址:/^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/
            HTML 标签:/^<([a-z]+)([^<]+)*(?:>(.*)<\/\1>|\s+\/>)$/
            Unicode编码中的汉字范围:/^[u4e00-u9fa5],{0,}$/
            匹配中文字符的正则表达式:[\u4e00-\u9fa5]
            ''')


def str_extract_mailAddr(mail_str):
    '''
    func : 邮箱
    抽取字符串中的邮箱地址
    输入str返回一个list
    '''
    pattern = re.compile(r"[a-zA-Z0-9_-]+@[a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9_-]+)")
    res_list = pattern.findall(mail_str)
    return res_list


def str_extract_personal_ID(strs):
    '''
    func : 身份证号
    抽取字符串中的身份证号
    输入str返回一个list
    '''
    pattern = re.compile(
        r"[1-9]\d{5}(?:18|19|(?:[23]\d))\d{2}(?:(?:0[1-9])|(?:10|11|12))(?:(?:[0-2][1-9])|10|20|30|31)\d{3}[0-9Xx]"
    )
    res_list = pattern.findall(strs)
    return res_list


def str_extract_CN_cellPhoneNumber(strs):
    '''
    func : 国内手机号码
    抽取字符串中的国内手机号码
    输入str返回一个list
    '''
    pattern = re.compile(r"1[356789]\d{9}")
    res_list = pattern.findall(strs)
    return res_list


def str_extract_CN_phoneNumber(strs):
    '''
    func : 国内固定电话
    抽取字符串中的国内固定电话
    输入str返回一个list
    '''
    pattern = re.compile(r"\d{3}-\d{8}|\d{4}-\d{7}")
    res_list = pattern.findall(strs)
    return res_list


def str_extract_webSite(strs):
    '''
    func : 域名
    抽取字符串中的网站域名
    输入str返回一个list
    '''
    pattern = re.compile(
        r"(?:(?:http:\/\/)|(?:https:\/\/))?(?:[\w](?:[\w\-]{0,61}[\w])?\.)+[a-zA-Z]{2,6}(?:\/)"
    )
    res_list = pattern.findall(strs)
    return res_list


def str_extract_IP_addr(strs):
    '''
    func : IP地址
    抽取字符串中的IP地址
    输入str返回一个list
    '''
    pattern = re.compile(
        r"((?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d))"
    )
    res_list = pattern.findall(strs)
    return res_list


def str_extract_datetime(strs):
    '''
    func : 日期
    抽取字符串中的日期yyyyMMdd、yyyy-MM-dd、yyyy/MM/dd、yyyy.MM.dd
    输入str返回一个list
    '''
    pattern = re.compile(r"\d{4}(?:-|\/|.)\d{1,2}(?:-|\/|.)\d{1,2}")
    res_list = pattern.findall(strs)
    return res_list


def str_extract_postcode(strs):
    '''
    func : 国内邮政编码
    抽取字符串中的国内邮政编码四级六位数编码结构-前两位数字表示省（直辖市、自治区）-第三位数字表示邮区；第四位数字表示县（市）-最后两位数字表示投递局（所）
    输入str返回一个list
    '''
    pattern = re.compile(r"[1-9]\d{5}(?!\d)")
    res_list = pattern.findall(strs)
    return res_list


def str_extract_password(strs):
    '''
    func : 密码
    抽取字符串中的密码(以字母开头，长度在6~18之间，只能包含字母、数字和下划线)
    输入str返回一个list
    '''
    pattern = re.compile(r"[a-zA-Z](?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,10}")
    res_list = pattern.findall(strs)
    return res_list


def str_extract_chinese(strs):
    '''
    func : 中文字符
    抽取字符串中的中文字符，分割成字符级别list
    输入str返回一个list
    '''
    pattern = re.compile(r"[\u4e00-\u9fa5]")
    res_list = pattern.findall(strs)
    return res_list


'''part_7 : pytorch_utility'''


def sentence_get_BERT_Token_online(samples):
    '''
    func : 得到句子的BERT_Token
    输入 : 中文的字符串sentence
    输出 : 一个Token_list
    '''
    from pytorch_transformers import BertTokenizer
    model_name = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokenized_text = [tokenizer.tokenize(i) for i in samples]
    input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]
    return input_ids


def sentence_get_BERT_Token_local(samples, maxlen):
    '''
    func : 得到句子的BERT_Token
    输入 : samples - 中文的字符串sentence合成的一个list或者array, maxlen - 想要的向量最大长度
    输出 : 一个二维list的input_ids
    '''
    from transformers import BertTokenizer
    from tqdm import tqdm
    bert_path = "D:/kwCodes/pretrained_model/chinese_roberta_wwm_ext_pytorch/"  # 该文件夹下存放三个文件（'vocab.txt', 'pytorch_model.bin', 'config.json'）
    tokenizer = BertTokenizer.from_pretrained(bert_path,
                                              local_files_only=True)  # 初始化分词器
    input_ids = []
    for line in tqdm(enumerate(samples)):
        encode_dict = tokenizer.encode_plus(text=line,
                                            max_length=maxlen,
                                            padding='max_length',
                                            truncation=True)
        input_ids.append(encode_dict['input_ids'])
    return input_ids


'''part_8 : machine learning utils'''


def set_seed(seed):
    '''跑实验便于复现，指定随机数'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu


# 编写自定义指标 - 二分类
# 精确率评价指标
def Precision(y_true, y_pred):
    TP = tf.reduce_sum(y_true * tf.round(y_pred))
    TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
    FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
    FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
    precision = TP / (TP + FP)
    return precision


# 召回率评价指标
def Recall(y_true, y_pred):
    TP = tf.reduce_sum(y_true * tf.round(y_pred))
    TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
    FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
    FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
    recall = TP / (TP + FN)
    return recall


# F1-score评价指标
def F1_score(y_true, y_pred):
    TP = tf.reduce_sum(y_true * tf.round(y_pred))
    TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
    FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
    FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1score = 2 * precision * recall / (precision + recall)
    return F1score


def main():
    pass


if __name__ == '__main__':
    main()
