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
  num_atoms = atom_types.shape[0]
  num_atom_features = count_ones_in_dict(atom_features_bool)
  num_pairs_features = count_ones_in_dict(pairs_features_bool)
  num_timesteps = get_num_timesteps(foldername)
  return num_atoms, num_atom_features, num_pairs_features, num_timesteps

def get_constants_bucket(foldername, atom_features_bool, pairs_features_bool):
  # Get some useful constants from the simulation
  client = boto3.client('s3')
  s3 = boto3.resource('s3')
  bucket = s3.Bucket('chemical-datasets')
  file_object = bucket.Object(key=foldername + 'input_atom/atom_types.npy').get()
  atom_types = np.load(BytesIO(file_object['Body'].read()))  
#   atom_types = np.load(foldername + 'input_atom/atom_types.npy')
  num_atoms = atom_types.shape[0]
  num_atom_features = count_ones_in_dict(atom_features_bool)
  num_pairs_features = count_ones_in_dict(pairs_features_bool)
  num_timesteps = get_num_timesteps_bucket(foldername, bucket)
  return num_atoms, num_atom_features, num_pairs_features, num_timesteps

def count_ones_in_dict(dictionary):
  num_ones = 0
  for key in dictionary.keys():
    if dictionary[key] == 1:
      num_ones += 1
  return num_ones

def get_num_timesteps(foldername):
  # For the simulation in foldername, get the number of timesteps, which is equal to
  # the number of files in output
  with h5py.File(foldername + 'output/output.hdf5', 'r') as f:
    num_of_keys = 0
    for key in f.keys():
      num_of_keys += 1
  # onlyfiles = next(os.walk(foldername + 'output/'))[2] #dir is your directory path as string
  return num_of_keys - 1 

def get_num_timesteps_bucket(foldername, bucket):
  # For the simulation in foldername, get the number of timesteps, which is equal to
  # the number of files in output
  file_object = bucket.Object(key=foldername + 'output/output.hdf5').get()
  with h5py.File(BytesIO(file_object['Body'].read()), 'r') as f:
    num_of_keys = 0
    for key in f.keys():
      num_of_keys += 1
  # onlyfiles = next(os.walk(foldername + 'output/'))[2] #dir is your directory path as string
  return num_of_keys - 1 

def keep_only_dictionary_with_iterations(dictionary, num_iterations):
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
