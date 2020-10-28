import boto3
import h5py
import numpy as np
from scipy import sparse
import tensorflow as tf
from io import BytesIO

class DataLoaderBucket():
    def __init__(self, foldername, atom_features_bool, pairs_features_bool,
                num_atom_features, num_pairs_features, num_atoms, 
                molecule_size_normalizer, cycle_size_normalizer, max_num_of_bonds,
                num_timesteps, validation_percentage):
        client = boto3.client('s3')
        s3 = boto3.resource('s3')
        self.bucket = s3.Bucket('chemical-datasets')
        self.foldername = foldername
        self.atom_features_bool = atom_features_bool
        self.pairs_features_bool = pairs_features_bool
        self.num_atom_features = num_atom_features
        self.num_pairs_features = num_pairs_features
        self.num_atoms = num_atoms
        self.molecule_size_normalizer = molecule_size_normalizer
        self.cycle_size_normalizer = cycle_size_normalizer
        self.max_num_of_bonds = max_num_of_bonds
        self.num_timesteps = num_timesteps
        self.validation_percentage = validation_percentage

    def get_data_with_generator(self, num_examples, batch_size):
        frame = 0
        while frame < num_examples:
          input_atom, input_pairs, atom_graph, mask, outputs = self.get_data(frame)
          atom_graph[:, :, 0] = frame % batch_size
          input_atom = tf.cast(input_atom, tf.float32)
          input_pairs = tf.cast(input_pairs, tf.float32)
          atom_graph = tf.cast(atom_graph, tf.int32)
          mask = tf.cast(mask, tf.float32)
          outputs = tf.cast(outputs, tf.float32)
          inputs = (input_atom, input_pairs, atom_graph, mask)    
          yield inputs, outputs
          frame += 1
        print(frame)

    def get_data(self, frame):
        pos_input_atom = 0
        input_atom = np.zeros([self.num_atoms, self.num_atom_features])
        if self.atom_features_bool["Atom types"] == 1:
          input_atom[:, pos_input_atom] = self.load_npy('input_atom/atom_types.npy') - 1
          #input_atom[:, pos_input_atom] = np.load(self.foldername + 'input_atom/atom_types.npy')-1
          pos_input_atom += 1
        if self.atom_features_bool["Cycle size"] == 1:
            input_atom = self.fill_with_hdf5_with_indices_1D(input_atom, pos_input_atom, 'input_atom/cycle_size.hdf5', frame, self.cycle_size_normalizer)
#           with h5py.File(self.foldername + 'input_atom/cycle_size.hdf5', 'r') as f:
#             temp = f['frame_' + str(frame)]
#             input_atom[temp[:, 0], pos_input_atom] = temp[:, 1]/self.cycle_size_normalizer
            pos_input_atom += 1
        if self.atom_features_bool["Molecule size"] == 1:
            input_atom = self.fill_with_hdf5_with_values_1D(input_atom, pos_input_atom, 'input_atom/molecule_size.hdf5', frame, self.molecule_size_normalizer)
#           with h5py.File(self.foldername + 'input_atom/molecule_size.hdf5', 'r') as f:
#             input_atom[:, pos_input_atom] = f['frame_' + str(frame)][()]/self.molecule_size_normalizer
            pos_input_atom += 1

        input_pairs = np.zeros([self.num_atoms, self.num_atoms, self.num_atom_features])
        pos_input_pairs = 0
        if self.pairs_features_bool["Bonded"] == 1:
            input_pairs = self.fill_with_hdf5_with_indices_2D(input_pairs, pos_input_pairs, 'input_pairs/bonded.hdf5', frame, 1)
#           with h5py.File(self.foldername + 'input_pairs/bonded.hdf5', 'r') as f:
#             temp = f['frame_' + str(frame)]
#             input_pairs[temp[:, 0], temp[:, 1], pos_input_pairs] = temp[:, 2]
            pos_input_pairs += 1
        if self.pairs_features_bool["Same cycle"] == 1:
            input_pairs = self.fill_with_hdf5_with_indices_2D(input_pairs, pos_input_pairs, 'input_pairs/same_cycle.hdf5', frame, 1)
#           with h5py.File(self.foldername + 'input_pairs/same_cycle.hdf5', 'r') as f:
#             temp = f['frame_' + str(frame)]
#             input_pairs[temp[:, 0], temp[:, 1], pos_input_pairs] = temp[:, 2]
            pos_input_pairs += 1
        if self.pairs_features_bool["Same molecule"] == 1:
            input_pairs = self.fill_with_hdf5_with_indices_2D(input_pairs, pos_input_pairs, 'input_pairs/same_molecule.hdf5', frame, self.molecule_size_normalizer)
#           with h5py.File(self.foldername + 'input_pairs/same_molecule.hdf5', 'r') as f:
#             temp = f['frame_' + str(frame)]
#             input_pairs[temp[:, 0], temp[:, 1], pos_input_pairs] = temp[:, 2]/self.molecule_size_normalizer
            pos_input_pairs += 1
        
        atom_graph = np.zeros([self.num_atoms, self.max_num_of_bonds, 2])
        atom_graph = self.fill_with_hdf5_with_indices_2D(atom_graph, 0, 'graph_info/atom_graph_1.hdf5', frame, 1)
        atom_graph = self.fill_with_hdf5_with_indices_2D(atom_graph, 1, 'graph_info/atom_graph_2.hdf5', frame, 1)
#         with h5py.File(self.foldername + 'graph_info/atom_graph_1.hdf5', 'r') as f:
#           temp = f['frame_' + str(frame)]
#           atom_graph[temp[:, 0], temp[:, 1], 0] = temp[:, 2]
#         with h5py.File(self.foldername + 'graph_info/atom_graph_2.hdf5', 'r') as f:
#           temp = f['frame_' + str(frame)]
#           atom_graph[temp[:, 0], temp[:, 1], 1] = temp[:, 2]
        mask = np.zeros([self.num_atoms, self.max_num_of_bonds, 1])
        if frame > 0:
            mask[np.where(atom_graph[:, :, 0]), 0] = 1
        else:
            mask[np.where(atom_graph[:, :, 1]), 0] = 1
            bonded_to_zero = np.array(atom_graph[0, np.where(atom_graph[0, :, 1])[0], 1], dtype=int)
            for at in bonded_to_zero:
                for at_bonded in range(self.max_num_of_bonds):
                    if atom_graph[at, at_bonded, 1] == 0:
                        mask[at, at_bonded, 0] = 1
                        break
        
        outputs = np.zeros([self.num_atoms, self.num_atoms])
        outputs = self.fill_with_hdf5_with_indices_2D_output(outputs, 'output/output.hdf5', frame, 1)
#         with h5py.File(self.foldername + 'output/output.hdf5', 'r') as f:
#           temp = f['frame_' + str(frame)]
#           outputs[temp[:, 0], temp[:, 1]] = temp[:, 2]
        
        return input_atom, input_pairs, atom_graph, mask, outputs

    def get_data_no_generator(self, num_examples, first_examples_bool, train_or_val):
      if first_examples_bool == 1:
        if train_or_val == 'train':
          examples_IDs = np.arange(num_examples)
        elif train_or_val == 'val':
          examples_IDs = np.arange((1 - self.validation_percentage) * self.num_timesteps, (1 - self.validation_percentage) * self.num_timesteps + num_examples)
      else:
        if train_or_val == 'train':
          examples_IDs = np.random.choice(np.arange((1-self.validation_percentage) * self.num_timesteps), size=num_examples, replace=False)
        elif train_or_val == 'val':
          examples_IDs = np.random.choice(np.arange((1-self.validation_percentage) * self.num_timesteps, self.num_timesteps), size=num_examples, replace=False)
      
      input_atom = np.zeros([num_examples, self.num_atoms, self.num_atom_features])
      input_pairs = np.zeros([num_examples, self.num_atoms, self.num_atoms, self.num_pairs_features])
      atom_graph = np.zeros([num_examples, self.num_atoms, self.max_num_of_bonds, 2])
      mask = np.zeros([num_examples, self.num_atoms, self.max_num_of_bonds, 1])
      outputs = np.zeros([num_examples, self.num_atoms, self.num_atoms])
      for i, frame in enumerate(examples_IDs):
        input_atom[i], input_pairs[i], atom_graph[i], mask[i], outputs[i] = self.get_data(i)

      input_atom = tf.cast(input_atom, tf.float32)
      input_pairs = tf.cast(input_pairs, tf.float32)
      atom_graph = tf.cast(atom_graph, tf.int32)
      mask = tf.cast(mask, tf.float32)
      outputs = tf.cast(outputs, tf.float32)
      return input_atom, input_pairs, atom_graph, outputs

    def load_npy(self, file_name):
        file_object = self.bucket.Object(key=self.foldername + file_name).get()
        npy_to_load = np.load(BytesIO(file_object['Body'].read()))  
        return npy_to_load
    
    def fill_with_hdf5_with_indices_1D(self, array_to_fill, last_dim_idx, file_name, frame, normalizer):
        file_object = self.bucket.Object(key=self.foldername + file_name).get()
        with h5py.File(BytesIO(file_object['Body'].read()), 'r') as f:
            temp = f['frame_' + str(frame)]
            array_to_fill[temp[:, 0], last_dim_idx] = temp[:, 1]/normalizer
        return array_to_fill
    
    def fill_with_hdf5_with_values_1D(self, array_to_fill, last_dim_idx, file_name, frame, normalizer):
        file_object = self.bucket.Object(key=self.foldername + file_name).get()
        with h5py.File(BytesIO(file_object['Body'].read()), 'r') as f:
            array_to_fill[:, last_dim_idx] = f['frame_' + str(frame)][()]/normalizer
        return array_to_fill
    
    def fill_with_hdf5_with_indices_2D(self, array_to_fill, last_dim_idx, file_name, frame, normalizer):
        file_object = self.bucket.Object(key=self.foldername + file_name).get()
        with h5py.File(BytesIO(file_object['Body'].read()), 'r') as f:
            temp = f['frame_' + str(frame)]
            array_to_fill[temp[:, 0], temp[:, 1], last_dim_idx] = temp[:, 2]/normalizer
        return array_to_fill
    
    def fill_with_hdf5_with_indices_2D_output(self, array_to_fill, file_name, frame, normalizer):
        file_object = self.bucket.Object(key=self.foldername + file_name).get()
        with h5py.File(BytesIO(file_object['Body'].read()), 'r') as f:
            temp = f['frame_' + str(frame)]
            array_to_fill[temp[:, 0], temp[:, 1]] = temp[:, 2]/normalizer
        return array_to_fill
