import tensorflow as tf
import numpy as np
import h5py

class StressStrainDS:

    def __init__(self, MAX_STRAIN=0.02, NUM_STRAINS=10, N_SAMPLES=60000):
        self.MAX_STRAIN = MAX_STRAIN
        self.NUM_STRAINS = NUM_STRAINS
        self.N_SAMPLES = N_SAMPLES

    def get_stress(self, strains, E, s_y, H):
        e_y = s_y / E
        elastic_strains = strains.copy()
        elastic_strains[elastic_strains > e_y] = e_y
        plastic_strains = strains - elastic_strains
        stresses = elastic_strains*E + plastic_strains*H
        return stresses

    def generate_samples(self, max_strain, n_strain, n_samples):
        strain = np.linspace(0, max_strain, n_strain + 1)[1:]
        stresses = np.empty((n_samples, n_strain))
        for i in range(n_samples):
            E = np.random.normal(1000, 50)
            s_y = np.random.normal(10, 0.5)
            H = np.random.normal(50, 5)
            stresses[i] = self.get_stress(strain, E, s_y, H)
        return stresses, strain

    def generate_dataset(self):
        stresses, strains = self.generate_samples(self.MAX_STRAIN, self.NUM_STRAINS, self.N_SAMPLES)
        stresses = np.array(stresses)
        
        with h5py.File('./datasets/stress_strain.hdf5', 'w') as f:
            f.create_dataset("stress_curves", data=stresses)
            f.create_dataset("strains", data=strains)

    def load_dataset(self):
        try:
            with h5py.File('./datasets/stress_strain.hdf5', 'r') as f:
                train_dataset = f['stress_curves'][:]
                
        except OSError:
            self.generate_dataset()

            with h5py.File('./datasets/stress_strain.hdf5', 'r') as f:
                train_dataset = f['stress_curves'][:]

        return train_dataset

    def load_strains(self):
        with h5py.File('./datasets/stress_strain.hdf5', 'r') as f:
            strains = f['strains']

        return strains