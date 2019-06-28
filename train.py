import os
import time
import numpy as np
from argparse import ArgumentParser
import tensorflow as tf

import gans
from data_generation import mnist, stress_strain
from utils import preprocessing, plotting

if __name__ == '__main__':
    parser = ArgumentParser(description = "GANs for stress-strain curve generation.")
    parser.add_argument('-m', '--model', default='dnn_1d', type=str)
    parser.add_argument('-d', '--dataset', default='stress_strain', type=str)
    parser.add_argument('-l', '--loss', default='wgan_gp', type=str)
    parser.add_argument('-opt', '--optimizer', default='adam', type=str)
    parser.add_argument('-e', '--epochs', default=50, type=int)
    parser.add_argument('-bs', '--batch_size', default=256, type=int) # Remember to automatically set in case of default
    parser.add_argument('-ci','--critic_iter', default=5, type=int)
    #parser.add_argument('-ci','--critic_iter', nargs='+', default=[0], type=int)
    parser.add_argument('-la','--LAMBDA', default=0.1, type=float)
    #parser.add_argument('-la','--lambda', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('-nd','--noise_dim', default=3, type=int)

    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('-b1', '--beta1', default=0.5, type=float)
    parser.add_argument('-b2', '--beta2', default=0.9, type=float)

    parser.add_argument('-s', '--save', default = True, type=bool)
    parser.add_argument('--result_dir', default = './results')
    
    args = parser.parse_args()

    # Model directory
    model_dir = args.result_dir + '/models/' + args.model + '/' + str(args.dataset) +'/BS_' + str(args.batch_size) + '/CI_' + str(args.critic_iter) + '/ND_' + str(args.noise_dim) + '/L_' + str(args.LAMBDA)

    # Load and prepare data
    if args.dataset.lower() == 'stress_strain':
        dataset = stress_strain.StressStrainDS()
        preproc = preprocessing.standardize
    elif args.dataset.lower() == 'mnist':
        dataset = mnist.MNISTDS()
        preproc = preprocessing.standardize_MNIST

    train_dataset = dataset.load_dataset()
    train_dataset, scaler = preprocessing.preprocess(train_dataset, args.batch_size, preproc)

    INPUT_SHAPE = tuple(tf.compat.v1.data.get_output_shapes(train_dataset).as_list()[1:])

    # Instantiate Generator and Discriminator
    generator, discriminator = gans.get_models(args.model, args.loss, INPUT_SHAPE, args.noise_dim)

    print('\n\n######### GENERATOR #########\n')
    generator.summary()
    print('\n\n####### DISCRIMINATOR #######\n')
    discriminator.summary()

    # Optimizers
    if args.optimizer.lower() == 'adam':
        generator_optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate, beta_1=args.beta1, beta_2=args.beta2)
        discriminator_optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate, beta_1=args.beta1, beta_2=args.beta2)

    elif args.optimizer.lower() == 'rmsprop':
        generator_optimizer = tf.keras.optimizers.RMSprop(lr=args.lr)
        discriminator_optimizer = tf.keras.optimizers.RMSprop(lr=args.lr)

    # Create Network
    gan = gans.GAN(discriminator, generator, discriminator_optimizer, generator_optimizer, args.loss, model_dir)
    
    # Train
    gan.train(train_dataset, args.epochs, args.batch_size, args.noise_dim, args.LAMBDA, args.critic_iter)
