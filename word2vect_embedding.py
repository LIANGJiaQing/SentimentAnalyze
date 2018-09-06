# -*- coding: utf-8 -*-
"""
Create on Wed Aug 29 15:36:50 2018

@author: Jiaqi Chen
"""

import numpy as np
import tensorflow as tf
import random
import collections
from collections import Counter
import jieba

from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.sans-serif'] = ['SimHei']    #用来正常显示中文标签
mpl.rcParams['font.family'] = 'STSong'
mpl.rcParams['font.size'] = 20

training_file = "hpscorepos_1_change_change.txt"

# 用CBOW模型进行词向量的处理
# 读取中文字,并解码成gb2312
def get_ch_label(txt_file):
    labels = ""
#    with open(txt_file, 'gbk') as f:
#        for label in f:
#            labels = labels+label.decode('gb2312') #解码成gb2312
    with open(txt_file, mode='r', encoding='utf-8') as f:
        for label in f:
            labels += label
    return labels

# 分词
def fenci(training_data):
    seg_list = jieba.cut(training_data)    #默认是精准模式
    training_ci = " ".join(seg_list)        #变成列表
    training_ci = training_ci.split()      #用空格将字符串分开
    training_ci = np.array(training_ci)
    training_ci = np.reshape(training_ci, [-1, ])
    return training_ci                      #返回列表

# 建立数据集
def build_dataset(words, n_words):
    """将原始输入处理到数据集中"""
    # 将统计词频0号位置给unknown
    # 其余按照词频由高到低排列
    # unknown获取按照预设词典大小350，词频排序靠后于350的都视为unknown

    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1)) #扩展列表，计算n_words个最常见的词的词频
    dictionary = dict()   #定义一个空字典
    for word, _ in count:  #word 是词语, _为不使用的临时变量
        dictionary[word] = len(dictionary)  #初始化字典每个词的词频的映射
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else: #在词典中的则保存相应列表位置词语的词频，否则为0
            index = 0        #dictionary['UNK'] unk的列表序号
            unk_count += 1     #unk word加1
        data.append(index)   #生成每个词的对应列表词频
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys())) #将词典转置，把k-v --> v--k
    print("反转词典类型是：",type(reversed_dictionary))
    return data, count, dictionary, reversed_dictionary

training_data = get_ch_label(training_file)
print("总字数", len(training_data))
training_ci = fenci(training_data)
print("总词数", len(training_ci))
training_label, count, dictionary, words = build_dataset(training_ci, 350)

words_size = len(dictionary)
print("字典词数", words_size)
print("Sample data样本数据", training_label[:10], [words[i] for i in training_label[:10]])

#############################
# 获取批次数据
# 取一定批次的样本数据，这一部分使用了CBOW模型构建样本
# 从开始位置的一个一个词作为输入，然后将其前面和后面的词作为标签，再分别组合在一起变成2组数据
# 对于一个中心词 在window范围 随机选取 num_skips个词，产生一系列的(input_id, output_id) 作为(batch_instance, label)
data_index = 0
def generate_batch(data, batch_size, num_skips, skip_window):
    """

    :param data: 数据集的label的每个词的词频
    :param batch_size: 每次训练时扫描的数据大小
    :param num_skips:表示input用了产生label的次数限制
    :param skip_window:窗口大小
    :return:批次数据和标签
    """
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1 #每一个样本由前skip_window + 当前target + 后skip_window组成

    buffer = collections.deque(maxlen=span)     #进程队列

    if data_index + span > len(data):
        data_index = 0

    buffer.extend(data[data_index : data_index + span])
    data_index += span

    for i in range(batch_size // num_skips):
        target = skip_window  #target在buffer中的索引为skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)

            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]

        if data_index == len(data):
            buffer = data[:span]
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    #注意防止越界
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

batch, labels = generate_batch(training_label, batch_size=8, num_skips=2, skip_window=1)
for i in range(8):     #先循环8次，然后组合好的样本与标签打印出来
    print(batch[i], words[batch[i]], "->", labels[i, 0], words[labels[i, 0]])

# 定义取样参数
batch_size = 128        #每批次取128个
embedding_size = 128    #embedding vector的维度（每个词向量的维度为128）
skip_window = 1         #左右的词数量
num_skips = 2           #一个input生成2个标签

valid_size = 16         #在0-word_size/2中的数取随机不能重复的16个字来验证模型
valid_window = np.int32(words_size/2)     #取样数据的分布范围
valid_examples = np.random.choice(valid_window, valid_size, replace=False)  #0-words_size/2中的数取16个。不能重复
num_sampled = 64        #负采样个数

# 定义模型变量
tf.reset_default_graph()

train_inputs = tf.placeholder(tf.int32, shape=[batch_size])         #输入占位符
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])      #标签占位符
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)      #验证数据占位符

# CPU上执行
with tf.device('/cpu:0'):
    #查找embedding
    # 94个词，每个128个向量
    embeddings = tf.Variable(tf.random_uniform([words_size,embedding_size], -1.0, 1.0))  #随机均匀正态分布化随机数，生成embedding
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    #计算NCE的loss的值
    #在反向传播中，embedding会与权重一起被nce_loss代表的loss值所优化更新，embeddings会一直改变
    nce_weights = tf.Variable(tf.truncated_normal([words_size, embedding_size], stddev=1.0/tf.sqrt(np.float32(embedding_size)))) #nce权重
    nce_biases = tf.Variable(tf.zeros([words_size]))    #nce偏差

#计算这批数据的平均损失值
# nce_loss会自动抽取每个负面标签的新样本
loss = tf.reduce_mean(
    tf.nn.nce_loss(weights=nce_weights,
                   biases=nce_biases,
                   labels=train_labels,
                   inputs=embed,
                   num_sampled=num_sampled,
                   num_classes=words_size))

# 梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

# 计算minibatch examples 和所有embeddings的cosine余弦相似度
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)     #找出对应向量的映射值
similiarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)  #计算出相似度
print("###############",similiarity.shape)

      
saver = tf.train.Saver()
savedir = "E:/AI/data/"
# 启动session,训练模型
num_steps = 500001
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("Initialized 初始化完毕")
    saver.save(sess, savedir+"word2vect_lijinhong_change.cpkt")
    
    average_loss = 0        #平均损失值
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(training_label, batch_size, num_skips, skip_window) #不断产生批次的数据进行训练
        feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

        _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val
        #通过打印测试可以看到。embed的值在逐渐被调节
        # emv = sess.run(embed, feed_dict={train_inputs: [37,18]})
        # print("emv---------------------", emv[0])

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            #平均loss
            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0

        #输入验证数据，显示结果
        if step % 10000 == 0:
            sim = similiarity.eval(session=sess)
            for i in range(valid_size):
                valid_word = words[valid_examples[i]]

                top_k = 8   #取排名最靠前的8个词
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]   #argsort函数返回的是数组值从小到大的索引值（sim是每个词和整个词典的夹角余弦）
                log_str = "Nearest to %s:" % valid_word

                for k in range(top_k):
                    close_word = words[nearest[k]]
                    log_str = '%s, %s' % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()

# 词向量可视化
def plot_with_labels(low_dim_embs, labels, filename='tsne3.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18,18)) #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x,y)
        plt.annotate(label,
                     xy=(x,y),
                     xytext=(5,2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(filename)

try:
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 300 #输出1000个词
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [words[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs, labels)
except ImportError:
    print('请安装sklearn、matplotlib和scipy以显示嵌入。')












































