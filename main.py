import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import time


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    image_input = tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

tests.test_load_vgg(load_vgg, tf)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # output_layer1 = tf.add(vgg_layer4_out, tf.layers.conv2d_transpose(vgg_layer7_out, 512, 2, 2))
    # output_layer2 = tf.add(vgg_layer3_out, tf.layers.conv2d_transpose(output_layer1, 256, 2, 2))
    # output_layer3 = tf.layers.conv2d_transpose(output_layer2, 128, 2, 2)
    # output_layer4 = tf.layers.conv2d_transpose(output_layer3, 64, 2, 2)
    # output_layer5 = tf.layers.conv2d_transpose(output_layer4, 2, 2, 2)
    #
    # return output_layer5

    # layer3_up = tf.layers.conv2d_transpose(vgg_layer3_out, 2, 8, 8)
    # layer4_up = tf.layers.conv2d_transpose(vgg_layer4_out, 2, 16, 16)
    # layer7_up = tf.layers.conv2d_transpose(vgg_layer7_out, 2, 32, 32)
    # output_layer = tf.add(layer7_up, tf.add(layer4_up, layer3_up))
    #
    # return output_layer

    layer7_up = tf.layers.conv2d_transpose(vgg_layer7_out, 512, 4, strides=(2,2), padding='same')
    layer4_up = tf.layers.conv2d_transpose(tf.add(vgg_layer4_out, layer7_up), 256, 4, strides=(2,2), padding='same')
    layer3_up = tf.layers.conv2d_transpose(tf.add(vgg_layer3_out, layer4_up), 2, 16, strides=(8, 8), padding='same')

    return layer3_up


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label)
    cross_entropy_loss = tf.reduce_mean(cross_entropy)
    train_op = tf.train.AdamOptimizer().minimize(cross_entropy_loss)


    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function

    for e in range(epochs):
        print("Epoch ", e)
        batches = get_batches_fn(batch_size)
        for X, y in batches:
            start = time.time()
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={ input_image: X, correct_label: y, keep_prob: 1.0, learning_rate: 0.01})
            print("loss ", loss, " in ", time.time() - start)

#tests.test_train_nn(train_nn)

def print_shapes(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, output_layer5,
           sess, batches_fn, input_image, correct_label, keep_prob, learning_rate):
    sess.run(tf.global_variables_initializer())
    for X, y in batches_fn(10):
        s1, s2, s3, s4 = sess.run(
            [tf.shape(vgg_layer3_out), tf.shape(vgg_layer4_out), tf.shape(vgg_layer7_out), tf.shape(output_layer5)],
            feed_dict={input_image: X, correct_label: y, keep_prob: 1.0, learning_rate: 0.01})
        print("Shapes - ", s1, ",", s2, ", ", s3, ", ", s4)


def run2():
    layer3 = tf.Variable(tf.ones((10, 20, 72, 256)))
    layer4 = tf.Variable(tf.ones((10, 10, 36, 512)))
    layer7 = tf.Variable(tf.ones((10, 5, 18, 4096)))

    layer7_up = tf.layers.conv2d_transpose(layer7, 512, 4, strides=(2,2), padding='same')
    layer4_up = tf.layers.conv2d_transpose(tf.add(layer4, layer7_up), 256, 4, strides=(2,2), padding='same')
    layer3_up = tf.layers.conv2d_transpose(tf.add(layer3, layer4_up), 2, 16, strides=(8, 8), padding='same')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        s = sess.run([tf.shape(layer7_up),tf.shape(layer4_up),tf.shape(layer3_up) ])
        print(s)

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/


    with tf.Session() as sess:

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        # get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/sample'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        correct_label = tf.placeholder(tf.float32, [None, 160, 576, 2])
        learning_rate = tf.placeholder(tf.float32, shape=[])

        # TODO: Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(last_layer, correct_label, learning_rate, num_classes)

        # print_shapes(layer3_out, layer4_out, layer7_out, last_layer, sess, get_batches_fn, image_input, correct_label,
        #             keep_prob, learning_rate)

        saver = tf.train.Saver()

        # TODO: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess, "./save2.ckpt")

        epochs = 5
        batch_size = 30
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input,
                 correct_label, keep_prob, learning_rate)

        save_path = saver.save(sess, "./new1.ckpt")
        print("Model saved in file: %s" % save_path)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
