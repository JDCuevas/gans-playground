import tensorflow as tf
from tensorflow.keras import layers

def get_loss_fns(loss_fn):
    
    def gan():
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        def gen_loss_fn(fake_output):
            loss = cross_entropy(tf.ones_like(fake_output), fake_output)
            return loss

        def disc_loss_fn(real_output, fake_output):
            real_loss = cross_entropy(tf.ones_like(real_output), real_output)
            fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
            total_loss = real_loss + fake_loss
            return total_loss

        return gen_loss_fn, disc_loss_fn

    def wgan_gp():

        def gen_loss_fn(fake_output):
            loss = -tf.reduce_mean(fake_output)
            return loss

        def disc_loss_fn(real_output, fake_output):
            real_loss = -tf.reduce_mean(real_output)
            fake_loss = tf.reduce_mean(fake_output) 
            total_loss = real_loss + fake_loss
            return total_loss

        return gen_loss_fn, disc_loss_fn

    if loss_fn.lower() == 'gan':
        generator_loss, discriminator_loss = gan()
    elif loss_fn.lower() == 'wgan_gp':
        generator_loss, discriminator_loss = wgan_gp()

    return generator_loss, discriminator_loss

def gradient_penalty(discriminator, real, fake, BATCH_SIZE):
    real = tf.cast(real, tf.float32)
    
    def _interpolate(a, b):
        alpha = tf.random.uniform(shape=[BATCH_SIZE, 1], minval=0., maxval=1.)
        inter = alpha * a + ((1 - alpha) * b)
        
        return inter

    x = _interpolate(real, fake)
    
    with tf.GradientTape() as t:
        t.watch(x)
        pred = discriminator(x)
        
    grad = t.gradient(pred, x)
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1]))
    gp = tf.reduce_mean((slopes - 1) ** 2)
    
    return gp

def _gradient_penalty(discriminator, real, fake, BATCH_SIZE):
    real = tf.cast(real, tf.float32)

    def _interpolate(a, b):
        shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
        alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
        inter = a + alpha * (b - a)
        inter.set_shape(a.shape)
        return inter

    x = _interpolate(real, fake)
    with tf.GradientTape() as t:
        t.watch(x)
        pred = discriminator(x)
    grad = t.gradient(pred, x)
    norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
    gp = tf.reduce_mean((norm - 1.)**2)

    return gp