# coding=utf-8
# 閫氳繃杈撳嚭top1 error鍜宼op10 error鏉ュ垽鏂缁冨拰妫€绱㈡晥�?
from __future__ import division
import tensorflow as tf
import numpy as np
import heapq
import resnet_model
import math
import oncequery as oq
import input


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('test_input', 'pathology_image/Test_data',
                           """Data input directory when using a product level model(trained).""")
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_classes', 4,
                            """Number of images to process in a batch.""")


MOVING_AVERAGE_DECAY = 0.9999


# 璇诲彇娴嬭瘯鍥剧�?
def readTestPic(filepath):

    # open a graph
    with tf.Graph().as_default() as g:
        # Build model(graph)

        trainset = input.ImageSet(FLAGS.test_input,False)
        images, labels, ids = trainset.next_batch(FLAGS.batch_size)
        # images, labels = cifar_input.build_input(
        # FLAGS.dataset, FLAGS.train_data_path, hps.batch_size, FLAGS.mode)
        hps = resnet_model.HParams(batch_size=FLAGS.batch_size,
                               num_classes=FLAGS.num_classes,
                               min_lrn_rate=0.0001,
                               lrn_rate=0.1,
                               num_residual_units=5,
                               use_bottleneck=False,
                               weight_decay_rate=0.0002,
                               relu_leakiness=0.1,
                               optimizer='mom')
        model = resnet_model.ResNet(hps, images, labels,'eval')
        model.build_graph()
        # Use our model
        logits = tf.get_default_graph().get_tensor_by_name("logit/xw_plus_b:0")
        print(logits)

        logits_norm = tf.nn.l2_normalize(logits, 1)

        # Run our model
        steps = math.ceil(
            trainset.num_exps / FLAGS.batch_size)  # *** Maybe exist some duplicate image features, next dict op will clear it.
        # Restore the moving average version of the learned variables for better effect.
        # for name in variables_to_restore:
        # 	print(name)
        saver = tf.train.Saver()


        with tf.Session() as sess:
        
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            logits_list = []
            id_list = []
            s = int(steps)
            for step in range(s):
                _ids,_logits= sess.run([ids, logits_norm])  # return nd-array
                put_2darray(_logits, logits_list)
                # if step == steps-1:
                # with open('G:/tmp/duplicate.txt','w') as f:
                # f.write(str(_ids.tolist()))
                for id in _ids.tolist():
                    id_list.append(id.decode('utf-8'))
                    
            coord.request_stop()
            coord.join(threads)
    return list(zip(id_list, logits_list))


def put_2darray(_2darray, li):
    _li = _2darray.tolist()
    for line in _li:
        li.append(line)


def calerrorrate(testfeatures, featurelib):
    totalfn = len(testfeatures)
    top1n = 0
    top10n = 0
    print(totalfn)
    for testfeature in testfeatures:
        qfeature = testfeature[1]
        dists = []

        for image_feature in list(featurelib.values()):
            dist = np.linalg.norm(np.array(qfeature) - np.array(image_feature))
            dists.append(dist)

        distlist = list(zip(list(featurelib.keys()), dists))
        top_k = heapq.nsmallest(10, distlist, key=lambda d: d[1])
        res = list(zip(*top_k))[0]

        resindex = []
        for re in res:
            index = int(re.split('#')[-2])
            resindex.append(index)

        qfeatureindex = int(testfeature[0].split('#')[-2])

        if qfeatureindex == resindex[0]:
            top1n += 1
            top10n += 1
        elif qfeatureindex in resindex:
            top10n += 1
        else:
            continue

        print('top1n:%d ,top10n:%d' % (top1n, top10n))
    
    
    print('top1nerrate: {:.2%}'.format(top1n/ totalfn))
    print('top10nerrate: {:.2%}'.format(top10n/ totalfn))
    top1n_errate = top1n / totalfn
    top10n_errate = top10n / totalfn
    return top1n_errate, top10n_errate


def main(argv=None):
    testfeatures = readTestPic(FLAGS.test_input)
    featurelib = oq.getFeaturelib()
    top1n_errate, top10n_errate = calerrorrate(testfeatures, featurelib)
    print("top1_errate: %.2f, top10_errate:%.2f" % (top1n_errate, top10n_errate))


if __name__ == '__main__':
    main()