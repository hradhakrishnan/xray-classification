
# coding: utf-8

# In[ ]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import sys
import tensorflow as tf
from PIL import Image,ImageFilter
import skimage.io as io
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
import numpy as np
import time

#example execution: python predict.py /Downloads/images_001/00000038_003.png

# In[ ]:

beginTime = time.time()


# In[ ]:
# Parameter definitions
batch_size = 10
learning_rate = 0.05
max_steps = 100
tfrecords_filename = 'image_prediction.tfrecords'
filenames = ['image_prediction.tfrecords']

img_height=256
img_width=256
# Object Classifications as Array
classes = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion','Emphysema','Fibrosis','Hernia',
                  'Infiltration','Mass','No Finding','Nodule','Pleural_Thickening','Pneumonia','Pneumothorax']

# In[103]:

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# In[ ]:
def _parse_record(tf_record):
    print('Parse Fn tf_record Called')
    features={'image_raw': tf.FixedLenFeature([], tf.string)}
    record = tf.parse_single_example(tf_record, features)
    image_raw = tf.decode_raw(record['image_raw'], tf.uint8)
    image_decoded = tf.reshape(image_raw, [-1,256,256])
    image_decoded = tf.reshape(image_decoded,[65536])
    return image_decoded



# testing using a Binary file, need not write input image to a binary tfrecord, the np array can be passed directly to predictclass
# for reshaping and passing to the model eval
def imagereshape(argv):
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    if os.path.exists(argv):
        img = Image.open(argv)
        img = img.resize((img_width, img_height), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        img.show()
        xray_img = np.array(img)
        img_raw = xray_img.tostring()
        height = xray_img.shape[0]
        width = xray_img.shape[1]
        print(xray_img.shape)

        example = tf.train.Example(features=tf.train.Features(feature={
          'image_raw': _bytes_feature(img_raw)}))

        writer.write(example.SerializeToString())
        print("writing tfrecord" + tfrecords_filename)
        writer.close()

        return argv


# In[ ]:
def predictclass(imgvalue):
    #------------------------------------------------------------
    # TFRECORD Dataset Iterator
    #------------------------------------------------------------
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_record)
    dataset = dataset.batch(1)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    images_placeholder = tf.placeholder(tf.float32, shape=[None,65536])
    labels_placeholder = tf.placeholder(tf.int64, shape=[None])

    weights = tf.Variable(tf.zeros([65536, 15]))
    biases = tf.Variable(tf.zeros([15]))
    tf.summary.histogram("weights", weights)
    tf.summary.histogram("biases", biases)

    logits = tf.matmul(images_placeholder, weights) + biases
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels_placeholder))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(logits, 1), labels_placeholder)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)

        saver.restore(sess, "trained_model.ckpt")

        sess.run(iterator.initializer)

        for i in range(1):
            image = sess.run(next_element)
            feed_dict = {images_placeholder: image}
            print('printing image')
            print(feed_dict)


            #print ("Model restored.")
            print("prediction happening here")
            prediction=tf.argmax(logits,1)
            return prediction.eval(feed_dict=feed_dict, session=sess)

# In[ ]:

def main(argv):
    """
    Main function.
    """
    imgvalue = imagereshape(argv)
    prediction = predictclass(imgvalue)
    print("-------------------------------------")
    print("MODELS PREDICTION")
    print("-------------------------------------")
    print(classes[prediction[0]])
    print("-------------------------------------")


if __name__ == "__main__":
    main(sys.argv[1])


# In[ ]:

endTime = time.time()
print('Total time: {:5.2f}s'.format(endTime - beginTime))
