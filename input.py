'''
Author: hiocde
Email: hiocde@gmail.com
Start: 3.3.17
Completion: 3.5.17
Original: Unfortunately, the former input.py is aborted by me , Yesterday I found it's not working and 
not a auto-pipeline in fact. However I have a new idea to build the pipeline now and 
I believe I'll succeed, And the interface for upper layer won't change.
--------------------
Now everything is OK.
Domain: main_dir{sub_dir{same class raw images}...}
'''
import os
import tensorflow as tf

# Global constants describing the Paris Dataset.
# A epoch is one cycle which train the whole training set.
# There are 20 corrupted images in Paris dataset.(VGG didn't delete them)
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 6412 - 20

class ImageSet:

    def __init__(self, img_main_dir, train=True):
        self.examples= self.create_examples(img_main_dir)		
        self.images_nm, self.labels= zip(*self.examples)	#separate images'name from label		
        self.num_exps= len(self.examples)
        self.train= train	# Training mode or use mode, default serving for training.


    def next_batch(self, batch_size):
        '''
            Return:
                imgs: 4d tensor, shape: [batch_size,h,w,d](float)
                labels: int32 1d tensor
                ids: string 1d tensor
        '''
        # Create a queue that produces the filenames to read.	
        if(self.train):
            filename_queue = tf.train.string_input_producer(self.images_nm, shuffle=self.train)
        else:
            filename_queue = tf.train.string_input_producer(self.images_nm, shuffle=self.train,num_epochs=1)
        img, label, id=self.read_example(filename_queue)

        img= self.distort_image(img)

        # Build batch queue.
        # capacity must be larger than min_after_dequeue and the amount determines the maximum we can prefetch.
        # Capacity recommendation: min_after_dequeue + (num_threads + a small safety margin) * batch_size
        # num_threads: The number of threads enqueuing tensors, ***a few shuffle effect because multi-thread***.
        min_rate_of_batch_queue = 0.4
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_rate_of_batch_queue)
        
        if self.train:
            imgs, labels, ids= tf.train.shuffle_batch(
                [img, label, id],
                batch_size=batch_size,
                num_threads=8,
                capacity=min_queue_examples + 12 * batch_size,
                min_after_dequeue=min_queue_examples)			
        else:
            imgs, labels, ids= tf.train.batch(
                [img, label, id],
                batch_size=batch_size,
                num_threads=8,
                capacity=min_queue_examples)
        print("----------------------------------")
        print(imgs.shape)
        print(labels)
        print("----------------------------------")
        return imgs, labels, ids

    def read_example(self, filename_queue):

        reader= tf.WholeFileReader()
        id, img_s= reader.read(filename_queue)	#path and image content(string tensor)
        #It's not convenient to use tensor trans such as split/slice, so add suffix "#label#" into subdir to make op easy.	
        label= tf.string_to_number(tf.string_split([id],'#').values[-2])
        try:
            img= tf.image.decode_jpeg(img_s,3) # It's enough
        except:
            print(tf.Session().run(id))
        # img= tf.image.decode_image(img_s,3)		# Better than decode_jpeg
        # decode_image output a tensor with no shape(But doc say no, it's weird) , it'll occur error when use tf.image.resize_images
        # img.set_shape([None, None, 3])
        
        return img, label, id

    def distort_image(self, img):
        '''
            Image preprocess.
        '''
        distorted_image = tf.cast(img, tf.float32)
        #It's fun that 224*224 input size in Alex's paper, but in fact is 227*227.( (224 - 11)/4 + 1 is quite clearly not an integer)
        distorted_image = tf.image.resize_image_with_crop_or_pad(
                        distorted_image, 224, 224)
        if self.train:
            distorted_image = tf.image.random_flip_left_right(distorted_image)
            distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)
            distorted_image = tf.image.random_contrast(distorted_image,lower=0.2,upper=1.8)
        return tf.image.per_image_standardization(distorted_image)

    def create_examples(self, img_main_dir):
        '''Args:
            img_main_dir: includes sub_dirs, each sub_dir is a class of images
            Return:
            all images path and their labels(0-n-1),a list of tuple
        '''
        examples=[]
        for sub_dir in os.listdir(img_main_dir):
            print(sub_dir)
            class_index= int(sub_dir.split('#')[-2])     # Must Append '#label_index#' to sub_dir
            print(class_index)
            for img_name in os.listdir(os.path.join(img_main_dir, sub_dir)):
                examples.append((os.path.join(img_main_dir,sub_dir,img_name), class_index))
        return examples
