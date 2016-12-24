import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Deconvolution2D, BatchNormalization, \
    Reshape, Convolution2D, Flatten
from keras.models import Model


def build_generator(batch_size, output_size=64,
                    output_channels=3, n_filters=64):
    latent_input = Input(shape=(100,), name='latent_input')
    # Prepare the convolution sizes
    s = output_size
    s2, s4 = int(np.ceil(s / 2.)), int(np.ceil(s / 4.))
    s8, s16 = int(np.ceil(s / 8.)), int(np.ceil(s / 16.))

    # Project the latent input into higher dimension to be reshaped
    projection = Dense(n_filters * 8 * s16 * s16,
                       activation='relu')(latent_input)
    projection = BatchNormalization(mode=2)(projection)
    # Reshape high dimension projection into num_filters x 4 x 4
    reshaped_projection = Reshape((n_filters * 8, s16, s16))(projection)
    # First deconvolution
    deconv1 = Deconvolution2D(nb_filter=n_filters * 4, nb_row=5, nb_col=5,
                              output_shape=(batch_size, n_filters * 4, s8, s8),
                              subsample=(2, 2), border_mode='same',
                              activation='relu')(reshaped_projection)
    deconv1 = BatchNormalization(mode=2, axis=1)(deconv1)
    # Second deconvolution
    deconv2 = Deconvolution2D(nb_filter=n_filters * 2, nb_row=5, nb_col=5,
                              output_shape=(batch_size, n_filters * 2, s4, s4),
                              subsample=(2, 2), border_mode='same',
                              activation='relu')(deconv1)
    deconv2 = BatchNormalization(mode=2, axis=1)(deconv2)
    # Third deconvolution
    deconv3 = Deconvolution2D(nb_filter=n_filters, nb_row=5, nb_col=5,
                              output_shape=(batch_size, n_filters, s2, s2),
                              subsample=(2, 2), border_mode='same',
                              activation='relu')(deconv2)
    deconv3 = BatchNormalization(mode=2, axis=1)(deconv3)
    # Fourth deconvolution
    deconv4 = Deconvolution2D(nb_filter=output_channels, nb_row=5, nb_col=5,
                              output_shape=(batch_size, output_channels, s, s),
                              subsample=(2, 2), border_mode='same',
                              activation='tanh')(deconv3)

    generator = Model(input=latent_input, output=deconv4)
    return generator


def build_discriminator(input_size=64, input_channels=3, n_filters=64):
    images = Input(shape=(input_channels, input_size, input_size),
                   name='images')
    conv = images
    for filter_mul in [1, 2, 4, 8]:
        conv = Convolution2D(nb_filter=n_filters * filter_mul,
                              nb_row=5, nb_col=5,
                              subsample=(2, 2), border_mode='same',
                              activation='relu')(conv)
        conv = BatchNormalization(mode=2, axis=1)(conv)

    # First convolution
    #conv1 = Convolution2D(nb_filter=n_filters, nb_row=5, nb_col=5,
    #                      subsample=(2, 2), border_mode='same',
    #                      activation='relu')(images)
    #print 'conv1', conv1.get_shape()
    #conv1 = BatchNormalization(mode=2, axis=1)(conv1)
    ## Second Convolution
    #conv2 = Convolution2D(nb_filter=n_filters * 2, nb_row=5, nb_col=5,
    #                      subsample=(2, 2), border_mode='same',
    #                      activation='relu')(conv1)
    #print 'conv2', conv2.get_shape()
    #conv2 = BatchNormalization(mode=2, axis=1)(conv2)
    ## Third Convolution
    #conv3 = Convolution2D(nb_filter=n_filters * 4, nb_row=5, nb_col=5,
    #                      subsample=(2, 2), border_mode='same',
    #                      activation='relu')(conv2)
    #print 'conv3', conv3.get_shape()
    #conv3 = BatchNormalization(mode=2, axis=1)(conv3)
    ## Fourth Convolution
    #conv4 = Convolution2D(nb_filter=n_filters * 8, nb_row=5, nb_col=5,
    #                      subsample=(2, 2), border_mode='same',
    #                      activation='relu')(conv3)
    #print 'conv4', conv4.get_shape()
    #conv4 = BatchNormalization(mode=2, axis=1)(conv4)
    # Classifier
    flattened = Flatten()(conv)
    classifier = Dense(1, activation='sigmoid')(flattened)

    discriminator = Model(input=images, output=classifier)
    return discriminator


def build_dgcan(image_size=64, image_channels=3, batch_size=64, n_filters=64):
    s = image_size
    latent_input = tf.placeholder(tf.float32, (batch_size, 100),
                                  name='latent_input')
    images = tf.placeholder(tf.float32, (batch_size, image_channels, s, s),
                            name='images')
    with tf.variable_scope('generator'):
        generator = build_generator(batch_size,
                                    output_size=image_size,
                                    output_channels=image_channels,
                                    n_filters=n_filters)
    with tf.variable_scope('discriminator'):
        discriminator = build_discriminator(input_size=image_size,
                                            input_channels=image_channels,
                                            n_filters=n_filters)

    # Generating images
    generated_images = generator(latent_input)

    # Discriminate both real and fake images
    real_discrimination = discriminator(images)
    fake_discrimination = discriminator(generated_images)

    # Create losses. 0 - fake / 1 - real
    # We punish the discriminator to predict that a fake image was true
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        fake_discrimination, tf.zeros_like(fake_discrimination)))
    # We punish the discriminator to predict that a true image was fake
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        real_discrimination, tf.ones_like(real_discrimination)))
    # The total loss of the discriminator is the sum of the above
    d_loss = d_loss_fake + d_loss_real
    # We punish the generator for creating images that did not fool the
    # discriminator
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        fake_discrimination, tf.ones_like(fake_discrimination)))

    return generator, discriminator, generated_images, d_loss, g_loss


def train_dgcan(stream, config):
    with tf.Session() as sess:
        generator, discriminator, generated_images, d_loss, g_loss = \
            build_dgcan(batch_size=config.batch_size,
                        image_size=config.image_size,
                        image_channels=config.image_channels)

        # Retrieve variables from discriminator and generator
        generator_vars = tf.get_collection(tf.GraphKeys().GLOBAL_VARIABLES,
                                           'generator')
        discriminator_vars = tf.get_collection(tf.GraphKeys().GLOBAL_VARIABLES,
                                               'discriminator')

        # Create optimizers
        optimizer = tf.train.AdamOptimizer(config.learning_rate,
                                           beta1=config.beta1)
        d_optim = optimizer.minimize(d_loss, var_list=discriminator_vars)
        g_optim = optimizer.minimize(g_loss, var_list=generator_vars)

        # Initialize all variables
        tf.global_variables_initializer().run()

        # Create saver
        saver = tf.train.Saver()

        iteration = 0
        for epoch in range(config.epochs):
            epoch_iterator = stream.get_epoch_iterator()
            while True:
                try:
                    # Get batch of real images
                    batch_images = next(epoch_iterator)[0]
                except StopIteration:
                    break
                iteration += 1
                # Create batch of latent z vectors
                batch_z = np.random.uniform(
                    -1, 1, [config.batch_size, 100]).astype(np.float32)

                # Run optimization of discriminator once
                sess.run(d_optim, feed_dict={'latent_input:0': batch_z,
                                             'images:0': batch_images})

                # Run optization of generator twice, as advised here:
                # https://github.com/Newmu/dcgan_code
                sess.run(g_optim, feed_dict={'latent_input:0': batch_z})
                _, images = sess.run([g_optim, generated_images],
                                     feed_dict={'latent_input:0': batch_z})

                if (iteration % 10) == 0:
                    sys.stdout.write('\rIteration {}'.format(iteration))
                    sys.stdout.flush()
                    plt.imsave('gen_im_it{}.jpeg'.format(iteration), images[0])

                if (iteration % 100) == 0:
                    plt.imsave('gen_im_it{}.jpeg'.format(iteration), images[0])


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    class Config(object):
        pass

    config = Config()
    config.epochs = 10
    config.batch_size = 32
    config.learning_rate = 0.0002
    config.beta1 = .5
    # build_dgcan(1)
    # train_dgcan(None, config)

    def test_generator():
        z = np.random.rand(1, 100)

        generator = build_generator(1)
        output = generator.outputs[0]
        print type(output)
        discriminator = build_discriminator()
        image = generator.predict(z, batch_size=1)

        print image.shape
        reshaped_image = np.rollaxis(image[0], 2, 1)
        print reshaped_image.shape
        print reshaped_image.min(), reshaped_image.max(), reshaped_image.mean()

        is_real = discriminator.predict(image, batch_size=1)
        print is_real

        plt.figure()
        plt.imshow(reshaped_image[0])
        plt.show()

    test_generator()
