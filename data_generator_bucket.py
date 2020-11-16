import boto3
import h5py
import numpy as np
from scipy import sparse
import tensorflow as tf
from io import BytesIO

# This class allows to load the data with a generator or not from Amazon S3 bucket.

class DataLoaderBucket():
    def __init__(self, foldername, atom_features_bool, pairs_features_bool,
                num_atom_features, num_pairs_features, num_atoms, 
                molecule_size_normalizer, cycle_size_normalizer, max_num_of_bonds,
                num_timesteps, validation_percentage):
        client = boto3.client('s3') # Get the S3 bucket
        s3 = boto3.resource('s3')
        self.bucket = s3.Bucket('chemical-datasets') # Get the folder 'chemical-datasets' in the bucket
        self.foldername = foldername # Foldername from where to get the data
        self.atom_features_bool = atom_features_bool  # Dictionary giving which features for the atoms to use
        self.pairs_features_bool = pairs_features_bool  # Dictionary giving which features for the atom pairs to use
        self.num_atom_features = num_atom_features  # Number of features for each atom
        self.num_pairs_features = num_pairs_features  # Number of features for each atom pairs
        self.num_atoms = num_atoms  # Number of atoms in the simulation
        self.molecule_size_normalizer = molecule_size_normalizer
        self.cycle_size_normalizer = cycle_size_normalizer
        self.max_num_of_bonds = max_num_of_bonds  # Max number of bonds a single atom can form
        self.num_timesteps = num_timesteps  # Number of timesteps in the simulation
        self.validation_percentage = validation_percentage  # Percentage of the frames that will be in the validation set
        self.extract_pairs = self.get_extract_pairs()

    def get_extract_pairs(self):
        extract_pairs = np.zeros([self.num_atoms*(self.num_atoms - 1)//2, 3], dtype=int)
        pos = 0
        for i in range(self.num_atoms - 1):
          for j in range(i + 1, self.num_atoms):
            extract_pairs[pos, 0] = 0
            extract_pairs[pos, 1] = i
            extract_pairs[pos, 2] = j
            pos += 1
        return extract_pairs

    def get_data_with_generator(self, num_examples, batch_size):
        # Define a generator to load the data.
        frame = 1
        while frame < num_examples:
          input_atom, input_pairs, atom_graph, mask, outputs, output_time = self.get_data(frame)
          atom_graph[:, :, 0] = (frame - 1) % batch_size
          extract_pairs = self.extract_pairs.copy()
          extract_pairs[:, 0] = (frame - 1) % batch_size
          input_atom = tf.cast(input_atom, tf.float32)
          input_pairs = tf.cast(input_pairs, tf.float32)
          input_pairs = tf.gather_nd(input_pairs, self.extract_pairs[:, 1:])
          atom_graph = tf.cast(atom_graph, tf.int32)
          mask = tf.cast(mask, tf.float32)
          extract_pairs = tf.cast(extract_pairs, tf.int32)
          outputs = tf.cast(outputs, tf.float32)
          outputs = tf.gather_nd(outputs, self.extract_pairs[:, 1:])
          output_time = tf.cast(output_time, tf.float32)
          inputs = (input_atom, input_pairs, atom_graph, mask, extract_pairs)    
          outputs_all = (outputs, output_time)
          yield inputs, outputs_all
          frame += 1
        print(frame)

    def get_data(self, frame):
        # Get the data for a specific frame
        pos_input_atom = 0
        input_atom = np.zeros([self.num_atoms, self.num_atom_features])  # All the features for each atom
        # Check if we want to use all atom features and load them if yes.
        if self.atom_features_bool["Atom types"] == 1:
          input_atom[:, pos_input_atom] = self.load_npy('input_atom/atom_types.npy') - 1
          pos_input_atom += 1
        if self.atom_features_bool["Cycle size"] == 1:
            input_atom = self.fill_with_hdf5_with_indices_1D(input_atom, pos_input_atom, 'input_atom/cycle_size.hdf5', frame, self.cycle_size_normalizer)
            pos_input_atom += 1
        if self.atom_features_bool["Molecule size"] == 1:
            input_atom = self.fill_with_hdf5_with_values_1D(input_atom, pos_input_atom, 'input_atom/molecule_size.hdf5', frame, self.molecule_size_normalizer)
            pos_input_atom += 1

        input_pairs = np.zeros([self.num_atoms, self.num_atoms, self.num_atom_features]) # All the features for each atom pair
        pos_input_pairs = 0
        # Check if we want to use all atom pairs features and load them if yes.
        if self.pairs_features_bool["Bonded"] == 1:
            input_pairs = self.fill_with_hdf5_with_indices_2D(input_pairs, pos_input_pairs, 'input_pairs/bonded.hdf5', frame, 1)
            pos_input_pairs += 1
        if self.pairs_features_bool["Same cycle"] == 1:
            input_pairs = self.fill_with_hdf5_with_indices_2D(input_pairs, pos_input_pairs, 'input_pairs/same_cycle.hdf5', frame, 1)
            pos_input_pairs += 1
        if self.pairs_features_bool["Same molecule"] == 1:
            input_pairs = self.fill_with_hdf5_with_indices_2D(input_pairs, pos_input_pairs, 'input_pairs/same_molecule.hdf5', frame, self.molecule_size_normalizer)
            pos_input_pairs += 1
        
        atom_graph = np.zeros([self.num_atoms, self.max_num_of_bonds, 2]) # Gives the atom indices to which each atom is bonded.
        atom_graph = self.fill_with_hdf5_with_indices_2D(atom_graph, 0, 'graph_info/atom_graph_1.hdf5', frame, 1)
        atom_graph = self.fill_with_hdf5_with_indices_2D(atom_graph, 1, 'graph_info/atom_graph_2.hdf5', frame, 1)

        # Define mask from atom_graph. atom_graph has self.max_num_of_bonds possible bonds, however most of the atoms
        # form less than self.max_num_of_bonds, so we mask where there are less bonds.
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
        
        # The output is 1 at output[i,j] is output[i,j] reacts and 0 otherwise.
        outputs = np.zeros([self.num_atoms, self.num_atoms])
        outputs = self.fill_with_hdf5_with_indices_2D_output(outputs, 'output/output.hdf5', frame, 1)
        output_time = np.zeros([1])
        file_object = self.bucket.Object(key=self.foldername + 'output/output_time.hdf5').get()
        with h5py.File(BytesIO(file_object['Body'].read()), 'r') as f:
            temp = f['frame_' + str(frame)]
            output_time[0] = temp[()]*0.012

        return input_atom, input_pairs, atom_graph, mask, outputs, output_time

    def get_data_no_generator(self, num_examples, first_examples_bool, train_or_val):
        # Load all the data in arrays.  
      if first_examples_bool == 1: # Define if you want to keep only the first examples, or if you choose the examples randomly.
        if train_or_val == 'train':
          examples_IDs = np.arange(int(num_examples)) + 1
        elif train_or_val == 'val':
          # For the validation set, you only consider the last validation_percentage of the frames.
          examples_IDs = np.arange(int((1 - self.validation_percentage) * self.num_timesteps), int((1 - self.validation_percentage) * self.num_timesteps + num_examples)) + 1
      else:
        if train_or_val == 'train':
          examples_IDs = np.random.choice(np.arange(int((1-self.validation_percentage) * self.num_timesteps)), size=num_examples, replace=False) + 1
        elif train_or_val == 'val':
          examples_IDs = np.random.choice(np.arange(int((1-self.validation_percentage) * self.num_timesteps), int(self.num_timesteps)), size=num_examples, replace=False) + 1
      
      input_atom = np.zeros([num_examples, self.num_atoms, self.num_atom_features])
      input_pairs = np.zeros([num_examples, self.num_atoms, self.num_atoms, self.num_pairs_features])
      atom_graph = np.zeros([num_examples, self.num_atoms, self.max_num_of_bonds, 2])
      mask = np.zeros([num_examples, self.num_atoms, self.max_num_of_bonds, 1])
      extract_pairs = np.repeat(np.expand_dims(self.extract_pairs, axis=0), num_examples, axis=0)
      outputs = np.zeros([num_examples, self.num_atoms, self.num_atoms])
      output_time = np.zeros([num_examples])
      for i, frame in enumerate(examples_IDs):
        input_atom[i], input_pairs[i], atom_graph[i], mask[i], outputs[i], output_time[i] = self.get_data(frame)
        atom_graph[i, :, :, 0] = i
        extract_pairs[i, :, 0] = i


      input_atom = tf.cast(input_atom, tf.float32)
      input_pairs = tf.cast(input_pairs, tf.float32)
      atom_graph = tf.cast(atom_graph, tf.int32)
      mask = tf.cast(mask, tf.float32)
      extract_pairs = tf.cast(extract_pairs, tf.int32)
      input_pairs = tf.gather_nd(input_pairs, extract_pairs)
      outputs = tf.cast(outputs, tf.float32)
      outputs = tf.gather_nd(outputs, extract_pairs)
      output_time = tf.cast(output_time, tf.float32)
      return input_atom, input_pairs, atom_graph, outputs, output_time

    def load_npy(self, file_name):
        # Load npy file from the bucket.
        file_object = self.bucket.Object(key=self.foldername + file_name).get()
        npy_to_load = np.load(BytesIO(file_object['Body'].read()))  
        return npy_to_load
    
    def fill_with_hdf5_with_indices_1D(self, array_to_fill, last_dim_idx, file_name, frame, normalizer):
        # Load files that are 1D and stored with only indices (sparse arrays)
        file_object = self.bucket.Object(key=self.foldername + file_name).get()
        with h5py.File(BytesIO(file_object['Body'].read()), 'r') as f:
            temp = f['frame_' + str(frame)]
            array_to_fill[temp[:, 0], last_dim_idx] = temp[:, 1]/normalizer
        return array_to_fill
    
    def fill_with_hdf5_with_values_1D(self, array_to_fill, last_dim_idx, file_name, frame, normalizer):
        # Load files that 1D and stored with values
        file_object = self.bucket.Object(key=self.foldername + file_name).get()
        with h5py.File(BytesIO(file_object['Body'].read()), 'r') as f:
            array_to_fill[:, last_dim_idx] = f['frame_' + str(frame)][()]/normalizer
        return array_to_fill
    
    def fill_with_hdf5_with_indices_2D(self, array_to_fill, last_dim_idx, file_name, frame, normalizer):
        # Load files that are 2D and stored with indices (sparse matrices)
        file_object = self.bucket.Object(key=self.foldername + file_name).get()
        with h5py.File(BytesIO(file_object['Body'].read()), 'r') as f:
            temp = f['frame_' + str(frame)]
            array_to_fill[temp[:, 0], temp[:, 1], last_dim_idx] = temp[:, 2]/normalizer
        return array_to_fill
    
    def fill_with_hdf5_with_indices_2D_output(self, array_to_fill, file_name, frame, normalizer):
        # Loaf the output files that are 2D and stored with indices (sparse matrices).
        file_object = self.bucket.Object(key=self.foldername + file_name).get()
        with h5py.File(BytesIO(file_object['Body'].read()), 'r') as f:
            temp = f['frame_' + str(frame)]
            array_to_fill[temp[:, 0], temp[:, 1]] = temp[:, 2]/normalizer
        return array_to_fill
