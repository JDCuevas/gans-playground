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