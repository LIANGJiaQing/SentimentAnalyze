# coding=utf-8

import jieba
import os
from nltk.parse.stanford import StanfordParser
from nltk.internals import find_jars_within_path

# 调整词典，可调节单个词语的词频，使其能（或不能）被分出来，调整分词效果
def adjust_jieba_dict(adjust_word_file):
    f = open(adjust_word_file, encoding='gbk')
    adjust_list = f.readlines()
    for i in adjust_list:
        jieba.suggest_freq(i.strip(), True)
    f.close()

# 中文分词，以空格隔开
def commend_segment(old_filename, new_filename):
    f = open(old_filename, encoding='utf-8')
    old_file_data = f.readlines()   # 读取旧文件的数据
    list_data = []                  # 全局变量存储新数据
    for i in old_file_data:         # 循环遍历旧文件数据
        data = ""
        seg_list = jieba.cut(i, cut_all=False, HMM=False) # 分词，生成迭代器
        print(data)
        for j in seg_list:   #循环输出迭代器
            data += j + " "
        print(data)
        list_data.append(data)
    f.close()
    write_list_to_file(list_data, new_filename)     #把新数据写会新文件，保留原文件

# # 把数据写回文件
def write_list_to_file(list_data, filename):
    f = open(filename, encoding='utf-8',mode='w')
    for i in list_data:
        f.write(i + '\n')
    f.close()


# 1,2,3三个方法处理句法树并写进文件，最后生成的临时文件要手动删掉
# 1-跳板文件，跳板文件名自己设置，自动生成，但结束后要自己手动删除
def write_to_temp_file(sentence_parse, filename):
    f = open(filename, encoding='utf-8', mode='w')
    f.write(sentence_parse + '\n')
    f.close()

# 2-返回跳板文件中的的句法树string文本
def return_str_tofile(filename):
    all_sentences = ""
    with open(filename, encoding='utf-8', mode='r') as f:
        sentences = f.readlines()
        for line in sentences:
            all_sentences += line.strip()
    return all_sentences

# 3-追加分析好的句子
def append_to_file(sentence_parse, filename):
    f = open(filename, encoding='utf-8', mode='a')
    f.write(sentence_parse + '\n')
    f.close()

# stanford_nlp句法分析
def stanford_parse_analyse(filename, new_filename, temp_filename):
    # 手动设置环境变量，此处要改，改成相应文件的路径
    JAVA_PATH = "D:/JAVA/bin/java.exe"
    os.environ['JAVAHOME'] = JAVA_PATH
    os.environ["STANFORD_PARSER"] = "E:/nltk/packages/stanford-parser/stanford-parser.jar"
    os.environ["STANFORD_MODELS"] = "E:/nltk/packages/stanford-parser/stanford-parser-3.6.0-models.jar"
    chinese_parser = StanfordParser(
        model_path='E:/nltk/packages/stanford-parser/edu/stanford/nlp/models/lexparser/chinesePCFG.ser.gz') #chineseFactored.ser.gz
    STANFORD_DIR = chinese_parser._classpath[0].rpartition('/')[0]
    chinese_parser._classpath = tuple(find_jars_within_path(STANFORD_DIR))
    # 设置java的JVM可容量的句子长度，默认是1000
    # chinese_parser.java_options = '-mx5000m'
    chinese_parser.java_options = '-mx11000m'
    # 循环输出，并写入新文件
    f = open(filename, encoding='utf-8', mode='r')
    data_list = f.readlines()
    # 循环读取
    index_flag = 1 #标志位，第几句话
    for i in data_list:
        if i.strip() != "":
            res = list(chinese_parser.parse((i.strip()).split()))
            print('#############第', index_flag, '评论句法分析结构如下##################')
            write_to_temp_file(filename=temp_filename, sentence_parse=str(res[0]))
            new_res = return_str_tofile(filename=temp_filename)
            append_to_file(sentence_parse=new_res, filename=new_filename)
            index_flag += 1
        else: pass
    f.close()

if __name__== '__main__':
    # 执行如下：
    # 1. 先adjust_jieba_dict("连接词文件路径...")（句法分析不需要停用词）
    # 2. 分词——commend_segment(old_filename(原csv文件名), new_filename（新的分词文件名，最好是txt文件）)
    # 3. 句法分析——stanford_parse_analyse(filename(分好词的文件路径txt文件), new_filename(句法分析生成的树string存放的文件), temp_filename(这个是临时文件，随便取))
    # 手动设置环境变量，此处要改，改成相应文件的路
    # 例如:::
    adjust_jieba_dict('connecting_words_find.txt')
    commend_segment('lenovoscoreneg_1.csv','lenovoscoreneg_2.txt')
    stanford_parse_analyse(filename='lenovoscoreneg_2.txt', new_filename='lenovoscoreneg_parse_3.txt',temp_filename='temp_file.txt')  #要处理一下，不能读取空行，会报错
 