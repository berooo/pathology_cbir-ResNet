#coding=utf-8
import tensorflow as tf
import resnet_model
import io
import json
import numpy as np
import heapq
import math
import matplotlib.pyplot as plt
from scipy import misc

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_dir', 'temp',
                    """Model checkpoint dir path.""")

tf.app.flags.DEFINE_string('feature_lib', 'output/logits_features.json',
                    """Model checkpoint dir path.""")
tf.app.flags.DEFINE_string('mode',
                           'eval',
                           'train or eval.')
IMG_SIZE=224
MOVING_AVERAGE_DECAY = 0.9999


def query(filepath,output,rank):
    #鎻愬彇寰呮悳绱㈠浘鐗囩殑鐗瑰�?
    qfeature=getQfeature(filepath)
    #璇诲彇鏁版嵁搴撲腑鐨勭壒寰佸瓧鍏?key鏄枃浠跺悕,value鏄壒寰佸€?
    featurelib=getFeaturelib()
    dists=[]

    for image_feature in list(featurelib.values()):
        dist=np.linalg.norm(np.array(qfeature)-np.array(image_feature))
        dists.append(dist)

    distlist=list(zip(list(featurelib.keys()),dists))
    top_k=heapq.nsmallest(rank,distlist, key= lambda d:d[1])
    res=list(zip(*top_k))[0]

    with open(output, 'w') as sf:
        sf.write(json.dumps(res))
        print('Write result ok!')

def getFeaturelib():
    featurelib=FLAGS.feature_lib
    with open(featurelib,'r') as f:
        featurelib_dict=json.loads(f.read())
        return featurelib_dict

def getQfeature(filepath):

        image=read(filepath)
        labels=[3]
        hps = resnet_model.HParams(batch_size=1,
                                   num_classes=4,
                                   min_lrn_rate=0.0001,
                                   lrn_rate=0.1,
                                   num_residual_units=5,
                                   use_bottleneck=False,
                                   weight_decay_rate=0.0002,
                                   relu_leakiness=0.1,
                                   optimizer='mom')
        model = resnet_model.ResNet(hps, image, labels, FLAGS.mode)
        model.build_graph()

        logits = tf.get_default_graph().get_tensor_by_name("logit/xw_plus_b:0")
        print(logits)

        logits_norm = tf.nn.l2_normalize(logits, 1)

        # Run our model
        steps = 1  # *** Maybe exist some duplicate image features, next dict op will clear it.
        # Restore the moving average version of the learned variables for better effect.
        # for name in variables_to_restore:
        # 	print(name)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            # Restore model from checkpoint.
            # Note!: checkpoint file not a single file, so don't use like this:
            # saver.restore(sess, '/path/to/model.ckpt-1000.index') xxx
            # Don't forget launch queue, use coordinator to avoid harmless 'Enqueue operation was cancelled ERROR'(of course you can also just start)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # ckpt correspond to 'checkpoint' file.
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            # model_checkpoint_path looks something like: /path/to/model.ckpt-1000
            print(ckpt.model_checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                # fc1_list=fc2_list=fc3_list=[] # the same object!
            logits_list = []
            _logits = sess.run([logits_norm])  # return nd-array
            print('................')
            print(_logits)
            print('................')
            put_2darray(_logits, logits_list)

            return logits_list


def read(filepath):
    I = misc.imread(filepath)
    distort_img=tf.cast(I,tf.float32)
    distorted_image = tf.image.resize_images(distort_img, [IMG_SIZE, IMG_SIZE])
    image=tf.image.per_image_standardization(distorted_image)
    img=tf.reshape(image,[1,IMG_SIZE,IMG_SIZE,3])
    #print(img)
    return img


def put_2darray(_2darray,li):
    #_li=_2darray.tolist()
    for line in _2darray:
        li.append(line)


def save(feature_list,file_nm):
    with io.open(file_nm,'w',encoding='utf-8') as file:
        file.write(json.dumps(feature_list, sort_keys=True))


def main(argv=None):
    query('Normal.jpg','n.txt',10)


if __name__ == '__main__':
	main()

