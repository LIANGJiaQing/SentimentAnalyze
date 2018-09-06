# coding=utf-8

import os
import sys
import textprocessing as tp
import numpy as np

projectPath = os.path.abspath(os.path.dirname(sys.argv[0]))

'''加载情感词典 start'''
# 正情感词典(清华大学)
tsinghuaPosDictPath = projectPath + "/dictionary/Tsinghua/tsinghua_positive_gb.txt"
tsinghuaPosDictData = tp.get_txt_data(tsinghuaPosDictPath, "lines")