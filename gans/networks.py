import os
import time
import csv
import functools
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from gans.losses import get_loss_fns, gradient_penalty, _gradient_penalty
from utils import plotting

class GAN():
    def __init__(self, discriminator, generator, disc_optimizer, gen_optimizer, loss, model_dir, dataset_name=None, seed=None):
        self.discriminator = discriminator
        self.generator = generator
        self.discriminator_optimizer = disc_optimizer
        self.generator_optimizer = gen_optimizer

        self.loss = loss
        gen_loss, disc_loss = get_loss_fns(loss)
        self.discriminator_loss = disc_loss
        self.generator_loss = gen_loss

        self.dataset_name = dataset_name
        self.seed = seed
        self.model_dir = model_dir

    @tf.function
    def disc_train_step(self, batch, BATCH_SIZE, NOISE_DIM, LAMBDA):
        with tf.GradientTape() as disc_tape:
            noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
            generated_batch = self.generator(noise, training=True)

            real_output = self.discriminator(batch, training=True)
            fake_output = self.discriminator(generated_batch, training=True)

            disc_loss = self.discriminator_loss(real_output, fake_output)

            if self.loss == 'wgan_gp':
                gp = _gradient_penalty(functools.partial(self.discriminator, training=True), batch, generated_batch, batch.shape[0])
                disc_loss += (gp * LAMBDA)

        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        
        return disc_loss

    @tf.function
    def gen_train_step(self, BATCH_SIZE, NOISE_DIM):
        with tf.GradientTape() as gen_tape:
            noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
            generated_batch = self.generator(noise, training=True)
            
            fake_output = self.discriminator(generated_batch, training=True)
            
            gen_loss = self.generator_loss(fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        return gen_loss

    def train(self, dataset, EPOCHS, BATCH_SIZE, NOISE_DIM, LAMBDA, CRITIC_ITERS, save_losses=True):
        if not os.path.exists(os.path.dirname(self.model_dir + '/')):
            try:
                os.makedirs(os.path.dirname(self.model_dir + '/'))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        
        checkpoint_dir = self.model_dir + '/training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                        discriminator_optimizer=self.discriminator_optimizer,
                                        generator=self.generator,
                                        discriminator=self.discriminator)
        
        losses_per_epoch = []
        
        for epoch in range(EPOCHS):
            start = time.time()
            iteration = 0
            bar = tqdm(total=60000)

            print('\nEpoch {}'.format(epoch + 1))

            for batch in tqdm(dataset):
                iter_start = time.time()
                disc_loss = self.disc_train_step(batch, BATCH_SIZE, NOISE_DIM, LAMBDA)
                
                if iteration % CRITIC_ITERS == 0:
                    gen_loss = self.gen_train_step(BATCH_SIZE, NOISE_DIM)
                    
                if (iteration + 1) % 50 == 0:
                        print('\tTime for iteration {} is {:.5f} sec'.format(iteration + 1, time.time() - iter_start))
                    
                # Updt progress bar
                bar.update(BATCH_SIZE)
                iteration += 1

            if (epoch + 1) % 10 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
                
            print('\nTime for epoch {} is {:.4f} sec'.format(epoch + 1, time.time() - start))
            print("Discriminator loss: {:.4f}\t Generator loss: {:.4f}\n".format(disc_loss.numpy(), gen_loss.numpy()))
            
            losses_per_epoch.append({'epoch' : epoch, 'discriminator loss' : disc_loss.numpy(), 'generator loss' : gen_loss.numpy()})

            # Generate images per epoch to viualize progress overtime
            if self.dataset_name == 'mnist':
                plotting.generate_and_save_images(self.generator, epoch + 1, self.seed, self.model_dir)

        # Generate image for last epoch
        if self.dataset_name == 'mnist':
                plotting.generate_and_save_images(self.generator, EPOCHS, self.seed, self.model_dir)

        # Save loss data
        if save_losses:
            with open(self.model_dir + '/losses_per_epoch.csv', mode='w') as csv_file:
                fieldnames = ['epoch', 'discriminator loss', 'generator loss']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

                writer.writeheader()
                writer.writerows(losses_per_epoch)
        
        