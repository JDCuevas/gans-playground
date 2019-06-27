import os
import time
import csv
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
    parser.add_argument('-bs', '--batch_size', nargs='+', default=[0], type=int) # Remember to automatically set in case of default
    parser.add_argument('-ci','--critic_iter', nargs='+', default=[0], type=int)
    parser.add_argument('-la','--lambda', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('-nd','--noise_dim', nargs='+', help='<Required> Set flag', required=True)

    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('-b1', '--beta1', default=0.5, type=float)
    parser.add_argument('-b2', '--beta2', default=0.9, type=float)

    parser.add_argument('-s', '--save', default = True, type=bool)
    parser.add_argument('--result_dir', default = './results/')
    
    args = parser.parse_args()


    # Load and prepare data
    if args.dataset.lower() == 'stress_strain':
        dataset = stress_strain.StressStrainDS()
    elif args.dataset.lower() == 'mnist':
        dataset = mnist.MNISTDS()

    train_dataset = dataset.

    INPUT_SHAPE = None

    # Instantiate Generator and Discriminator
    generator, discriminator = gans.get_models(args.model, args.loss, INPUT_SHAPE, args.noise_dim)

    print('######### GENERATOR #########')
    generator.summary()
    print('####### DISCRIMINATOR #######')
    discriminator.summary()

    # Optimizers
    if args.optimizer.lower() == 'adam':
        generator_optimizer = tf.keras.optimizers.Adam(lr=args.lr, beta_1=args.beta1, beta_2=args.beta2)
        discriminator_optimizer = tf.keras.optimizers.Adam(lr=args.lr, beta_1=args.beta1, beta_2=args.beta2)

    elif args.optimizer.lower() == 'rmsprop':
        generator_optimizer = tf.keras.optimizers.RMSprop(lr=args.lr)
        discriminator_optimizer = tf.keras.optimizers.RMSprop(lr=args.lr)
    
    
