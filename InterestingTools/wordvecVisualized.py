'''
输入 : 句子的vector，一个matrix就成
输出 : 输出句子的vector的聚类结果以及将PCA降维后的空间图片画出
'''
from time import time
import fanTools
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def sentence_get_BERT_Token_online(samples):
    '''
    func : 得到句子的BERT_Token
    输入 : 中文的字符串sentence
    输出 : 一个LongTensor类型的Token_list
    '''
    from pytorch_transformers import BertTokenizer
    model_name = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokenized_text = [tokenizer.tokenize(i) for i in samples]
    input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]
    input_ids = list(input_ids)
    return input_ids


def wordvecVisualized(words, n_clusters):
    embeddings = []
    for i in words:
        s_t = time()
        aa = list(sentence_get_BERT_Token_online(i))
        e_t = time() - s_t
        print('converting to word vectors : ', e_t)
        embeddings.append(aa)
    print("所有的词向量为 ： ",'\n',words,'\n',embeddings)
    embeds = []
    for i in embeddings:
        vec = []
        for j in i:
            vec.extend(j)
        embeds.append(vec)
    min_len = 999
    for i in embeds:
        if len(i) <= min_len:
            min_len = len(i)
    embeds_cut = [i[:min_len] for i in embeds]
    dim = PCA(n_components=2).fit_transform(embeds_cut)
    y_pred = KMeans(n_clusters=n_clusters).fit_predict(embeds_cut)
    classes = set([i for i in y_pred])
    print("聚类总类别数 : ", classes)
    cluster = {}
    for i, j in zip(y_pred, words):
        if i != -1:
            li = cluster.get(i, [])
            li.append(j)
            cluster[i] = li
    print("聚类结果为 : ", cluster)
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    plt.clf()
    plt.figure(figsize=(9, 9))
    plt.title('wordvec可视化结果')
    plt.scatter(dim[:, 0], dim[:, 1], c=y_pred)
    for w, (x, y) in zip(words, dim):
        plt.text(x + 0.5, y + 0.5, w, fontsize=15)
    plt.show()
    # plt.savefig(os.path.join(output, '%s.png' % file.strip('.txt')))


if __name__ == '__main__':
    words = ['您与我们订立的合同',
             '合同构成',
             '合同成立及生效',
             '投保年龄',
             '我们提供的保障',
             '保险金额',
             '保险期间',
             '保险责任',
             '癌症保险金',
             '责任免除']
    wordvecVisualized(words, 10)
