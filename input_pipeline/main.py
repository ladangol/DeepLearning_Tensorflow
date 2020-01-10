import tensorflow as tf
from glob import glob
from random import shuffle
import os

# Class representation of network model
class Model(object):

    # Initialize model
    def __init__(self, sess, learning_rate, batch_size, num_class, input_data_path, epoch_num):
        self.sess = sess
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.image_size = [64, 64]
        self.num_class = num_class
        self.input_data_path = input_data_path
        self.epoch_num = epoch_num

    def generate_file_list(self, in_path):
        data_path = []
        labels = []
        search_path = os.path.join(in_path, '*.jpg')
        files_list = glob(search_path)
        shuffle(files_list)
        for full_path in files_list:
            filename = os.path.basename(full_path)
            category = 1
            if 'cat' in filename:
                category = 0

            data_path.append(full_path)
            labels.append(category)

        return data_path, labels

    def _read_and_decode_image(self, inputs, labels):
        image = tf.read_file(inputs)
        image = tf.image.decode_image(image)
        image.set_shape([None, None, 3])
        image_resized = tf.image.resize_images(image, self.image_size)
        image_normalized = image_resized / 255.0
        return image_normalized, labels

    # Define loader for training dataset with mini-batch size 100
    def initialize_dataset(self):
        filenames,labels = self.generate_file_list(self.input_data_path)
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.map(self._read_and_decode_image)
        dataset = dataset.shuffle(buffer_size=500)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.batch_size * 5)
        iter = dataset.make_initializable_iterator()
        self.iterator = iter
        next = iter.get_next()
        return next

    # Define graph for model
    def build_model(self):

        # Define placeholders for input and ouput values
        self.x = tf.placeholder(tf.float32, [None, self.image_size[0],self.image_size[1],3], name='x')
        self.y = tf.placeholder(tf.int64, [None], name='y')

        # Define placeholder for learning rate
        self.learning_rt = tf.placeholder(tf.float32, name='learning_rt')

        h = tf.layers.conv2d(self.x, 32, (3,3), activation=tf.nn.relu)
        h = tf.layers.max_pooling2d(h,(2,2),1)

        h = tf.layers.conv2d(h, 32, (3,3), activation=tf.nn.relu)
        h = tf.layers.max_pooling2d(h,(2,2),1)

        h = tf.layers.flatten(h)

        # Define fully-connected layer with 20 hidden units
        h = tf.layers.dense(h, 50, activation=tf.nn.relu)

        # Define fully-connected layer to single ouput prediction
        self.logits = tf.layers.dense(h, self.num_class, activation=None)
        # Define loss function
        # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=self.logits)
        # Define optimizer
        self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rt).minimize(self.loss)
        # optimizer =tf.train.MomentumOptimizer(learning_rate=1e-1, momentum=0.7).minimize(cost)
        self.pred = tf.nn.softmax(self.logits)
        predictions = tf.argmax(self.pred, 1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y, predictions), tf.float32))
        # Define variable initializer
        self.init = tf.global_variables_initializer()

    # Train model
    def train(self):

        # Initialize variables
        self.sess.run(self.init)

        # Initialize dataset
        self.dataset = self.initialize_dataset()

        # Iterate through 20000 training steps
        for n in range(0, self.epoch_num):
            epoch_acc = 0
            num_iterate = 0
            self.sess.run(self.iterator.initializer)
            while True:
                try:
                    # Retrieve batch from loader for training dataset
                    x_batch, y_batch = self.sess.run(self.dataset)

                    # Apply decay to learning rate every 1000 steps
                    if n % 1000 == 0:
                        self.learning_rate = 0.9 * self.learning_rate

                    # Run optimization operation for current mini-batch
                    fd = {self.x: x_batch, self.y: y_batch, self.learning_rt: self.learning_rate}
                    opt,acc  = self.sess.run([self.optim,self.accuracy], feed_dict=fd)
                    print("Batch: ", num_iterate)
                    num_iterate += 1
                    epoch_acc += acc
                except tf.errors.OutOfRangeError:
                    break

            if num_iterate>0:
                print("epoch: " ,n,epoch_acc/num_iterate)

# Initialize and train model
def main():

    # Specify initial learning rate
    learning_rate = 0.001

    # Specify training batch size
    batch_size = 100

    num_class = 2

    input_data_path = "D:\\DeepLearning_Tensorflow\\dogs_vs_cats\\data\\train"

    epoch_num =10

    # Initialize TensorFlow session
    with tf.Session() as sess:
        # Initialize model
        model = Model(sess, learning_rate, batch_size, num_class, input_data_path, epoch_num)

        # Build model graph
        model.build_model()

        # Train model
        model.train()

# Run main() function when called directly
if __name__ == '__main__':
    main()