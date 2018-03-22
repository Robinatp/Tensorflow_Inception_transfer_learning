# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import  numpy as np
import PIL.Image as Image
from pylab import *
import time
from tensorflow.python.platform import gfile

# 下载模型
def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    output_tensor,input_tensor = tf.import_graph_def(graph_def, name='',
                        return_elements=["final_training_ops/Softmax:0","import/DecodeJpeg/contents:0"])
    with tf.Session() as sess:
        ops = sess.graph.get_operations()
        for op in ops:
            print(op.name)

  return graph, output_tensor,input_tensor

# 识别手势
def recognize(jpg_path, pb_file_path, classes):
  with tf.Graph().as_default():
      
      graph, output_tensor,input_tensor = load_graph(pb_file_path)

      
      with tf.Session(graph=graph) as sess:
#           # 获取输入张量
#           input_x = graph.get_tensor_by_name("import/DecodeJpeg/contents:0")
#           # 获取输出张量
#           output = graph.get_tensor_by_name("final_training_ops/Softmax:0")
          # 读入待识别图片
          image_data = gfile.FastGFile(jpg_path, 'rb').read()
          t1 = time.time()
          pre = sess.run(output_tensor, feed_dict={input_tensor:image_data})
          t2 = time.time()
          writer = tf.summary.FileWriter("./logs_inception_from_pb", graph=tf.get_default_graph())
          
          print(pre)
          results = np.squeeze(pre)
          prediction_labels = np.argmax(results, axis=0)
          classes =load_labels("tmp/retrained_labels.txt")
          top_k = results.argsort()[-5:][::-1]
          for i in top_k:
              print(classes[i], results[i])
        
          print('probability: %s: %.3g, running time: %.3g' % (classes[prediction_labels],results[prediction_labels], t2-t1))
          
          
def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

if __name__=="__main__":
  classes = load_labels("tmp/retrained_labels.txt")
  jpg_path = "flower_data/sunflowers/1022552002_2b93faf9e7_n.jpg"
  pb_file_path="tmp/frozen_graph.pb"
  recognize(jpg_path, pb_file_path, classes)