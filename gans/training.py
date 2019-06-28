def get_train_steps(loss):

    if loss.lower() == 'wgan_gp':
        @tf.function
        def disc_train_step(self, batch, NOISE_DIM, LAMBDA):
            with tf.GradientTape() as disc_tape:
                noise = tf.random.normal([batch.shape[0], NOISE_DIM])
                generated_batch = self.generator(noise, training=True)

                real_output = self.discriminator(batch, training=True)
                fake_output = self.discriminator(generated_batch, training=True)

                disc_loss = self.discriminator_loss(real_output, fake_output)
                gp = gradient_penalty(functools.partial(self.discriminator, training=True), batch, generated_batch, batch.shape[0])

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