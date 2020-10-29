import boto3
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import sparse
from io import BytesIO


def get_constants(foldername, atom_features_bool, pairs_features_bool):
  # Get some useful constants from the simulation
  atom_types = np.load(foldername + 'input_atom/atom_types.npy')
  num_atoms = atom_types.shape[0]  # Number of atoms in the simulations
  num_atom_features = count_ones_in_dict(atom_features_bool)  # Number of features to describe one atom
  num_pairs_features = count_ones_in_dict(pairs_features_bool)  # Number of features to describe an atom pair
  num_timesteps = get_num_timesteps(foldername)  # Number of timesteps in the simulation
  return num_atoms, num_atom_features, num_pairs_features, num_timesteps

def get_constants_bucket(foldername, atom_features_bool, pairs_features_bool):
  # Get some useful constants from the simulation when loading from Amazon S3 bucket.
  client = boto3.client('s3')  # Setting up the bucket
  s3 = boto3.resource('s3')
  bucket = s3.Bucket('chemical-datasets')  # Work with the folder 'chemical-datasets' of the bucket
  file_object = bucket.Object(key=foldername + 'input_atom/atom_types.npy').get()
  atom_types = np.load(BytesIO(file_object['Body'].read()))  
  num_atoms = atom_types.shape[0]   # Number of atoms in the simulations
  num_atom_features = count_ones_in_dict(atom_features_bool)  # Number of features to describe one atom
  num_pairs_features = count_ones_in_dict(pairs_features_bool)  # Number of features to describe an atom pair
  num_timesteps = get_num_timesteps_bucket(foldername, bucket)  # Number of timesteps in the simulation
  return num_atoms, num_atom_features, num_pairs_features, num_timesteps

def count_ones_in_dict(dictionary):
  # Get the number of values equal to 1 in a boolean dictionary.
  # Used for the boolean dictionary of features
  num_ones = 0
  for key in dictionary.keys():
    if dictionary[key] == 1:
      num_ones += 1
  return num_ones

def get_num_timesteps(foldername):
  # For the simulation in foldername, get the number of timesteps, which is equal to
  # the number of dataset in the HDF5 file of the output.
  with h5py.File(foldername + 'output/output.hdf5', 'r') as f:
    num_of_keys = 0
    for key in f.keys():
      num_of_keys += 1
  return num_of_keys - 1 

def get_num_timesteps_bucket(foldername, bucket):
  # For the simulation in foldername, get the number of timesteps, which is equal to
  # the number of dataset in the HDF5 file of the output. (Here for the bucket)
  file_object = bucket.Object(key=foldername + 'output/output.hdf5').get()
  with h5py.File(BytesIO(file_object['Body'].read()), 'r') as f:
    num_of_keys = 0
    for key in f.keys():
      num_of_keys += 1
  return num_of_keys - 1 

def keep_only_dictionary_with_iterations(dictionary, num_iterations):
  # The dictionary has the bond changes occuring at each timestep. This function
  # keeps only the keys in the dictionary before num_iterations reactions occur.
  new_dict = {}
  time_range = np.array([0])
  num_iterations_so_far = 0
  for i in dictionary.keys():
    new_dict[i] = dictionary[i]
    time_range = np.append(time_range, i)
    num_iterations_so_far += dictionary[i].shape[0]
    if num_iterations_so_far > num_iterations:
      break
  return new_dict, time_range

def plot_molecules_of_interest(molecules_per_frame, molecule_list, time_range, molecules_per_frame_MD, molecule_list_MD, time_range_MD, molecules_of_interest):
  # Plot the molecules_of_interest for the ground truth and the ML model results.
  fig_num = 0
  for i in range(molecules_of_interest.shape[0]):
    idx_mol = np.where((molecule_list == molecules_of_interest[i]).all(-1))[0][0]
    idx_mol_MD = np.where((molecule_list_MD == molecules_of_interest[i]).all(-1))[0][0]
    plt.figure(fig_num)
    plt.plot(time_range, molecules_per_frame[:, idx_mol])
    plt.title("ML model " + "C" + str(molecules_of_interest[i, 0]) + "H" + str(molecules_of_interest[i, 1]))
    plt.savefig("./plots/" + "ML_model_" + "C" + str(molecules_of_interest[i, 0]) + "H" + str(molecules_of_interest[i, 1]))
    fig_num += 1
    plt.figure(fig_num)
    plt.plot(time_range_MD, molecules_per_frame_MD[:, idx_mol_MD])
    plt.title("Ground truth " + "C" + str(molecules_of_interest[i, 0]) + "H" + str(molecules_of_interest[i, 1]))
    fig_num += 1
    plt.savefig("./plots/" + "truth_" + "C" + str(molecules_of_interest[i, 0]) + "H" + str(molecules_of_interest[i, 1]))
  longest_molecules = np.zeros(len(time_range))
  longest_molecules_MD = np.zeros(len(time_range_MD))
  for t in range(len(time_range)):
    mol_present = np.where(molecules_per_frame[t])
    longest_molecules[t] = np.max(molecule_list[mol_present, 0])
  for t in range(len(time_range_MD)):
    mol_present = np.where(molecules_per_frame_MD[t])
    longest_molecules_MD[t] = np.max(molecule_list_MD[mol_present, 0])
  plt.figure(fig_num)
  plt.plot(time_range, longest_molecules)
  plt.title("ML model, size of the longest molecule")
  plt.savefig("./plots/ML_model_longest_molecule")
  fig_num += 1
  plt.figure(fig_num)
  plt.plot(time_range_MD, longest_molecules_MD)
  plt.title("Ground truth, size of the longest molecule")
  plt.savefig("./plots/truth_longest_molecule")
  fig_num += 1
  plt.show()
