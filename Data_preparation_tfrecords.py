
# coding: utf-8

# In[101]:

import os
import csv
import tensorflow as tf
import numpy as np
from PIL import Image
import skimage.io as io
from skimage.transform import rescale, resize, downscale_local_mean



# In[102]:
# download and unzip the NIH image data and csv from https://nihcc.app.box.com/v/ChestXray-NIHCC
# Suggest to unzip the images to individual folders e.g: images001,images_002...images_012 etc. and create tfrecords for each batch
source_folderpath='/Images/images_001/'
tfrecords_filename = 'images_001.tfrecords'
img_height=256
img_width=256


# In[103]:

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# In[104]:

classes = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion','Emphysema','Fibrosis','Hernia',
                  'Infiltration','Mass','No Finding','Nodule','Pleural_Thickening','Pneumonia','Pneumothorax']
num_classes = len(classes)


# In[ ]:

print(classes[1])
print(classes.index('Nodule'))


# In[ ]:

#Convert images into tfrecords - Hari

writer = tf.python_io.TFRecordWriter(tfrecords_filename)

with open('Data_Entry_2017.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        classification_list = row['Finding Labels'].split('|')
        xray_image_path = os.path.join(source_folderpath, row['Image Index'])
        print(xray_image_path)
        for classification in classification_list:
            label = classes.index(classification)
            if os.path.exists(xray_image_path):
                img = Image.open(xray_image_path)
                img = img.resize((img_width, img_height), Image.ANTIALIAS)
                xray_img = np.array(img)
                img_raw = xray_img.tostring()
                height = xray_img.shape[0]
                width = xray_img.shape[1]
                print(xray_img.shape)

                example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(label),
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'image_raw': _bytes_feature(img_raw)}))

                writer.write(example.SerializeToString())
                print("writing tfrecord" + xray_image_path)


        print(row['Image Index'], row['Finding Labels'])
    writer.close()


# In[ ]:

# Below block of code if you need to validate the binary file to verify label, image and image size are recorded as intended
reconstructed_images = []

record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
output='<your folder path>/out.png'
for string_record in record_iterator:

    example = tf.train.Example()
    example.ParseFromString(string_record)

    height = int(example.features.feature['height']
                                 .int64_list
                                 .value[0])

    width = int(example.features.feature['width']
                                .int64_list
                                .value[0])

    img_string = (example.features.feature['image_raw']
                                  .bytes_list
                                  .value[0])

    label = int(example.features.feature['label']
                                .int64_list
                                .value[0])
    print(classes[label])

    img_1d = np.fromstring(img_string, dtype=np.uint8)
    reconstructed_img = img_1d.reshape((-1, width, height ))
    print(reconstructed_img.shape)
    print(reconstructed_img.dtype)
    io.imsave(output,reconstructed_img[0])
