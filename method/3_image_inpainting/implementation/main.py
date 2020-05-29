import tensorflow as tf
from models.DNN import *

# Datasets
######### Write directories of datasets
tf.app.flags.DEFINE_string('train_images_dir', '', 'Training images data directory.')
tf.app.flags.DEFINE_string('val_images_dir', '', 'Validation images data directory.')
tf.app.flags.DEFINE_string('test_images_dir', '', 'Testing images data directory.')
# tf.app.flags.DEFINE_string('vgg_path', '', 'vgg directory.')
 # 
tf.app.flags.DEFINE_boolean('train', True, 'whether to train the network')
tf.app.flags.DEFINE_integer('num_epochs', 7, 'epochs to train')
tf.app.flags.DEFINE_integer('train_batch_size', 1, 'number of elements in a training batch')
tf.app.flags.DEFINE_integer('val_batch_size', 1, 'number of elements in a validation batch')
tf.app.flags.DEFINE_integer('test_batch_size', 1, 'number of elements in a testing batch')

tf.app.flags.DEFINE_integer('height_of_image', 256, 'Height of the images.')
tf.app.flags.DEFINE_integer('width_of_image', 256, 'Width of the images.')
tf.app.flags.DEFINE_float('num_channels', 3, 'Number of the channels of the images.')

tf.app.flags.DEFINE_float('learning_rate', 0.00005, 'Learning rate of the optimizer')

tf.app.flags.DEFINE_integer('display_step', 1, 'Number of steps we cycle through before displaying detailed progress.')
tf.app.flags.DEFINE_integer('validation_step', 1, 'Number of steps we cycle through before validating the model.')
###### write somethong like "/Users/mariam/Desktop/ASDS/tez/autoencoder/results"
tf.app.flags.DEFINE_string('base_dir', '', 'Directory in which results will be stored.')
tf.app.flags.DEFINE_integer('checkpoint_step', 1, 'Number of steps we cycle through before saving checkpoint.')
tf.app.flags.DEFINE_integer('max_to_keep',2, 'Number of checkpoint files to keep.')

tf.app.flags.DEFINE_integer('summary_step', 2, 'Number of steps we cycle through before saving summary.')

tf.app.flags.DEFINE_string('model_name', 'inpainting', 'name of model')

FLAGS = tf.app.flags.FLAGS


def main(argv=None):
    model = DNN(
        train_images_dir=FLAGS.train_images_dir,
        val_images_dir=FLAGS.val_images_dir,
        test_images_dir=FLAGS.test_images_dir,
        num_epochs=FLAGS.num_epochs,
        train_batch_size=FLAGS.train_batch_size,
        val_batch_size=FLAGS.val_batch_size,
        test_batch_size=FLAGS.test_batch_size,
        height_of_image=FLAGS.height_of_image,
        width_of_image=FLAGS.width_of_image,
        num_channels=FLAGS.num_channels,
        # num_classes=FLAGS.num_classes,
        learning_rate=FLAGS.learning_rate,
        base_dir=FLAGS.base_dir,
        max_to_keep=FLAGS.max_to_keep,
        model_name=FLAGS.model_name,
        
    )


    if FLAGS.train:
        model.create_network(model_type="train")
        model.initialize_network()
        model.train_model(FLAGS.display_step, FLAGS.validation_step, FLAGS.checkpoint_step, FLAGS.summary_step)
    else:
        model.create_network(model_type="test")
        model.initialize_network()
        model.test_model()


if __name__ == "__main__":
    tf.app.run()
