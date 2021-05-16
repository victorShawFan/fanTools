'''
fanTools : Tools Created by xfan
For coding efficiency
fourth edition:
2021/05/16
'''
import collections
import json
import random
import re
import time
from difflib import SequenceMatcher

import jieba
import pymongo

# 因为太过简易使用所以暂时不写做func但又很好用的包：
# 1 - synonyms,

'''part_0 : 杂活王'''


def show_me_pycharm_shortcut():
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


import datetime


def now_when():
    """
    func : 输出现在的日期时间等
    输入 : 无
    输出 : str-现在的日期和时间
    """
    now = datetime.datetime.now()
    print(now)
    return str(now)[:-7]


'''part_1 : 数据文件读写与处理'''


# 家卿格式json数据的读写
# with open('dev.json', 'r', encoding='utf-8')as f:
#     lines = f.readlines()
#     for line in lines:
#         data = json.loads(line)  # 此时data为一个dict{text,spo_list:[]}
#         with open("train.json", 'a+',encoding='utf-8')as fout:
#             fout.write(("{}\n".format(json.dumps(data, ensure_ascii=False))))


def read_file_to_list(filepath, read_all=True, read_many=0, shuffle=False):
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
    with open(filename, mode='w', encoding='utf-8', ) as f:
        for di in data:
            print(json.dumps(di, ensure_ascii=False), file=f)


def append_json_lines(data, filename):
    """
    func : 将list追加入json
    输入 : data_list，文件路径读入文件
    输出 : 写入json
    """
    with open(filename, mode='a+', encoding='utf-8', ) as f:
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
    return THOUSANDS[num // 1000] + HUNDREDS[num % 1000 // 100] + TENS[num % 100 // 10] + ONES[num % 10]


def romanToInt(s):
    '''
    罗马数字转成数字
    输入 : 字符串 s - 含有古罗马数字
    输出 : 整数
    '''
    # 构建一个字典记录所有罗马数字子串，注意长度为2的子串记录的值是（实际值 - 子串内左边罗马数字代表的数值）
    d = {'I': 1, 'IV': 3, 'V': 5, 'IX': 8, 'X': 10, 'XL': 30, 'L': 50, 'XC': 80, 'C': 100, 'CD': 300, 'D': 500,
         'CM': 800, 'M': 1000}
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
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
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


def main():
    pass


if __name__ == '__main__':
    main()
