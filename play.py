'''
Author: hiocde
Email: hiocde@gmail.com
Start: 2.27.17
Completion: 3.10.17
Original: Extract images' features and output to json files.
Domain: Use a trained model to do something such as feature extract here.
'''

import tensorflow as tf
import json
import math
import io
#import sys
import input
import resnet_model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input', 'pathology_image/Training_data',
                            """Data input directory when using a product level model(trained and tested).""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'temp',
                    """Model checkpoint dir path.""")
tf.app.flags.DEFINE_string('output', 'output',
                   '''Model output dir, stores image features''')
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_classes', 4,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('mode',
                           'train',
                           'train or eval.')


def build_feature_lib():

    with tf.Graph().as_default() as g:
    # Build model(graph)
    # First build a input pipeline(Model's input subgraph).
        trainset = input.ImageSet(FLAGS.input)
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
        model = resnet_model.ResNet(hps, images, labels, FLAGS.mode)
        model.build_graph()

        tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        for tensor_name in tensor_name_list:
            print(tensor_name,'\n')
        logits=g.get_tensor_by_name("logit/xw_plus_b:0")
        print(logits)

        logits_norm=tf.nn.l2_normalize(logits,1)

        # Run our model
        steps= math.ceil(trainset.num_exps/FLAGS.batch_size) #*** Maybe exist some duplicate image features, next dict op will clear it.
        # Restore the moving average version of the learned variables for better effect.

        saver = tf.train.Saver()

        with tf.Session() as sess:
        # Restore model from checkpoint.
        # Note!: checkpoint file not a single file, so don't use like this:
        # saver.restore(sess, '/path/to/model.ckpt-1000.index') xxx
        # Don't forget launch queue, use coordinator to avoid harmless 'Enqueue operation was cancelled ERROR'(of course you can also just start)

            coord= tf.train.Coordinator()
            threads= tf.train.start_queue_runners(sess=sess, coord=coord)

        # ckpt correspond to 'checkpoint' file.
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        # model_checkpoint_path looks something like: /path/to/model.ckpt-1000 
            if ckpt and ckpt.model_checkpoint_path:       
                saver.restore(sess, ckpt.model_checkpoint_path)  
      
        # fc1_list=fc2_list=fc3_list=[] # the same object!
            logits_list=[]
            id_list=[]
            s=int(steps)
            print(steps)
            for step in xrange(s):
                _ids, _logits= sess.run([ids,logits_norm])	#return nd-array
                put_2darray(_logits, logits_list)
        # if step == steps-1:
        # with open('G:/tmp/duplicate.txt','w') as f:
        # f.write(str(_ids.tolist()))
                for id in _ids.tolist():
        # print(id) # id is byte string, not a valid key of json
                    id_list.append(id.decode('utf-8'))

                if step%10 == 0 or step == steps-1:
                    save(id_list, logits_list, FLAGS.output+'/logits_features.json')
                print('Step %d, %.3f%% extracted.'%(step, (step+1)/steps*100))

            coord.request_stop()
            coord.join(threads)

# def extract_batch_features():
# 	images, labels, ids = imageset.next_batch(FLAGS.batch_size)	# Dont need like alexnet.FLAGS.batch_size
# 	logits = alexnet.inference(images)
# 	softmax= tf.nn.softmax(logits)	#softmax = exp(logits) / reduce_sum(exp(logits), dim), dim=-1 means add along line.
# 	fc1= tf.get_default_graph().get_tensor_by_name("fc1:0")
# 	fc2= tf.get_default_graph().get_tensor_by_name("fc2:0")
# 	return fc1, fc2, softmax, ids

def put_2darray(_2darray, li):
    _li= _2darray.tolist()
    for line in _li:
        li.append(line)

def save(id_list, feature_list, file_nm):
    '''
    Save as json obj like {id:feature, ...}, A feature is a vector.
    Overwrite file if re-call.
    '''
    # Note! dict will auto-update duplicate key, so don't warry about extraction for the same image.
    dic = dict(list(zip(id_list, feature_list)))
    # mode='w' not 'a'
    with io.open(file_nm, 'w', encoding='utf-8') as file:
        # Note! dic does not keep sequence,
        # Use 'sort_keys'.(Of course it's does not matter if not use, just looks very chaotic)
        file.write(unicode(json.dumps(dic, sort_keys=True)))


def main(argv=None):
    build_feature_lib()

if __name__ == '__main__':	
    # ...About the argv parse, is a XuanXue.
    # I only know you can script it like this:
    # python3 play.py --argv1=val1 --argv2==val2 ...
    # to run it and the FLAGS is right.
    # if you're interested, to see the source of tf.app.run.

    tf.app.run()
    # or tf.app.run(argv=sys.argv)
    # or tf.app.run(build_feature_lib) PS:Add a useless arg in build_feature_lib.
    # or tf.app.run(build_feature_lib, argv=sys.argv)
    # All ok, so weird!