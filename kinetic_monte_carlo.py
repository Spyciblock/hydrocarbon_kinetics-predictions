import numpy as np
import networkx as nx
import tensorflow as tf

from data_generator import DataLoader

def run_KMC(model, num_iterations, foldername, atom_features_bool, pairs_features_bool,
                num_atom_features, num_pairs_features, num_atoms, 
                molecule_size_normalizer, cycle_size_normalizer, max_num_of_bonds,
                num_timesteps, validation_percentage):
  # Run the whole Kinetic Monte Carlo process using the trained "model"
  data_loader = DataLoader(foldername, atom_features_bool, pairs_features_bool,
                num_atom_features, num_pairs_features, num_atoms, 
                molecule_size_normalizer, cycle_size_normalizer, max_num_of_bonds,
                num_timesteps, validation_percentage)
  # Get the initial input for the model
  Xtest_input_atom, Xtest_input_pairs, Xtest_atom_graph, Xtest_mask, Ytest = data_loader.get_data_no_generator(1, 1, 'train')
  bond_change = {}
  first_frame = Xtest_input_pairs[0, :, :, 0].numpy()
  atom_types = Xtest_input_atom[0, :, 0]
  adjacency_matrix = first_frame.copy()
  time = [0]
  for i in range(num_iterations):
    if i % 10 == 0:
      print(i)
    # Get the "reactivity scores" from the model
    results = model.predict([Xtest_input_atom, Xtest_input_pairs, Xtest_atom_graph, Xtest_mask])
    # From the "reactivity scores" pick a reaction and the time before this reaction using the Kinetic Monte Carlo algorithm.
    adjacency_matrix, bond_change, time_new = run_step_KMC(results[0, :, :, 0], adjacency_matrix, bond_change, time[-1])
    time.append(time_new)
    # Update the system with the picked reaction and recalculate the input of the model.
    Xtest_input_atom, Xtest_input_pairs, Xtest_atom_graph = get_new_input(adjacency_matrix, atom_features_bool, pairs_features_bool, molecule_size_normalizer, cycle_size_normalizer, num_atoms, num_atom_features, num_pairs_features, max_num_of_bonds, atom_types)
  return bond_change, first_frame, time

def run_step_KMC(propensity, adjacency_matrix, bond_change, time):
  # Run one step of the Kinetic Monte Carlo. Choose the reaction to happen and the time before this
  # reaction using the propensities. Store these in bond_change and time
  propensity_temp = np.triu(propensity, k = 1).flatten()
  reaction_to_happen, time_to_reaction = pick_reaction_and_time(propensity_temp)
  at_1, at_2 = np.unravel_index(reaction_to_happen, adjacency_matrix.shape)
  adjacency_matrix[at_1, at_2] = 1 - adjacency_matrix[at_1, at_2]
  adjacency_matrix[at_2, at_1] = 1 - adjacency_matrix[at_2, at_1]
  time += time_to_reaction
  bond_change[time] = np.array([[at_1, at_2, adjacency_matrix[at_1, at_2]]])
  return adjacency_matrix, bond_change, time


def pick_reaction_and_time(propensity):
  # Choose the reaction that is going to happen and the delay before it happens.
  propensity_tot = np.sum(propensity)
  r = np.random.random((2,))
  time_to_reaction = 1/propensity_tot*np.log(1/r[0])
  reaction_to_happen = int(np.argwhere(np.cumsum(propensity) >= r[1]*propensity_tot)[0])
  return reaction_to_happen, time_to_reaction

def get_new_input(adjacency_matrix, atom_features_bool, pairs_features_bool, molecule_size_normalizer, cycle_size_normalizer, num_atoms, num_atoms_features, num_pairs_features, max_num_of_bonds, atom_types):
  # When one step of the Kinetic Monte Carlo process, all the input data for the 
  # model need to be computed from the new adjacency matrix.
  molecule_graph = nx.Graph(adjacency_matrix)  # Define the graph of the molecular system
  input_atom = get_input_atom(molecule_graph, atom_features_bool, molecule_size_normalizer, cycle_size_normalizer, num_atoms, num_atoms_features, atom_types)
  input_pairs = get_input_pairs(molecule_graph, adjacency_matrix, pairs_features_bool, molecule_size_normalizer, num_atoms, num_pairs_features)
  atom_graph, bond_graph, mask = get_graph_info(adjacency_matrix, num_atoms, max_num_of_bonds)
  return input_atom, input_pairs, atom_graph

def get_input_atom(molecule_graph, atom_features_bool, molecule_size_normalizer, cycle_size_normalizer, num_atoms, num_atoms_features, atom_types):
  # From molecule_graph, get the input_atom for the model.
  input_atom = np.zeros([1, num_atoms, num_atoms_features])
  pos = 0
  if atom_features_bool["Atom types"]:
    input_atom[0, :, pos] = atom_types
    pos += 1
  if atom_features_bool["Cycle size"]:
    input_atom[0, :, pos] = get_cycle_size(molecule_graph, num_atoms)/cycle_size_normalizer
    pos += 1
  if atom_features_bool["Molecule size"]:
    input_atom[0, :, pos] = get_molecule_size(molecule_graph, num_atoms)/molecule_size_normalizer
    pos += 1
  input_atom = tf.cast(input_atom, tf.float32)
  return input_atom

def get_cycle_size(molecule_graph, num_atoms):
  # Get the cycle size each atom is in.
  cycle_size = np.zeros([num_atoms])
  cycles = nx.cycle_basis(molecule_graph)
  for c in range(len(cycles)):
    num_of_atoms_in_cycles = len(cycles[c])
    for atom in cycles[c]:
      if cycle_size[atom] == 0:
          cycle_size[atom] = num_of_atoms_in_cycles
      else:
        if cycle_size[atom] > num_of_atoms_in_cycles:
            cycle_size[atom] = num_of_atoms_in_cycles
  return cycle_size

def get_molecule_size(molecule_graph, num_atoms):
  # Get the molecule size of the molecule containing each atom.
  molecule_size = np.zeros([num_atoms])
  molecules = nx.connected_components(molecule_graph)
  for mol in molecules:
      num_of_atoms_in_mol = len(mol)
      for atom in mol:
          molecule_size[atom] = num_of_atoms_in_mol
  return molecule_size

def get_input_pairs(molecule_graph, adjacency_matrix, pairs_features_bool, molecule_size_normalizer, num_atoms, num_pairs_features):
  # From molecule_graph, get the input_pairs for the ML model.
  input_pairs = np.zeros([1, num_atoms, num_atoms, num_pairs_features])
  pos = 0
  if pairs_features_bool["Bonded"] == 1:
    input_pairs[0, :, :, pos] = adjacency_matrix
    pos += 1
  if pairs_features_bool["Same cycle"] == 1:
    input_pairs[0, :, :, pos] = get_same_cycle(molecule_graph, num_atoms)
    pos += 1
  if pairs_features_bool["Same molecule"] == 1:
    input_pairs[0, :, :, pos] = get_same_molecule(molecule_graph, num_atoms)/molecule_size_normalizer
    pos += 1
  input_pairs = tf.cast(input_pairs, tf.float32)
  return input_pairs

def get_same_cycle(molecule_graph, num_atoms):
  # For a pair of atoms, same_cycle is 1 is the two atoms are in the same cycle and 0 otherwise.
  same_cycle = np.zeros([num_atoms, num_atoms])
  cycles = nx.cycle_basis(molecule_graph)
  for c in range(len(cycles)):
      atoms_in_cycles = []
      for atom in cycles[c]:
          atoms_in_cycles.append(atom)
      for i in range(len(atoms_in_cycles) - 1):
          for j in range(1, len(atoms_in_cycles)):
              idx_1 = atoms_in_cycles[i]
              idx_2 = atoms_in_cycles[j]
              same_cycle[idx_1, idx_2] = 1
              same_cycle[idx_2, idx_1] = 1
  return same_cycle

def get_same_molecule(molecule_graph, num_atoms):
  # For a pair of atoms, same_molecule is 0 is the two atoms are not in the same molecule and is equal
  # to the distance between these two atoms in the molecule otherwise.
  same_molecule = np.zeros([num_atoms, num_atoms])
  lengths = nx.all_pairs_shortest_path_length(molecule_graph)
  for i in lengths:
      atom = i[0]
      for atom_bonded in i[1].keys():
          same_molecule[atom, atom_bonded] = i[1][atom_bonded]
  return same_molecule

def get_graph_info(adjacency_matrix, num_atoms, max_num_of_bonds):
  # Get atom_graph and bond_graph inputs for the model
  atom_graph = np.zeros([1, num_atoms, max_num_of_bonds, 2])
  bond_graph = np.zeros([1, num_atoms, max_num_of_bonds, 3])
  mask = np.zeros([1, num_atoms, max_num_of_bonds])
  num_of_bonds_per_atom = np.zeros([num_atoms], dtype=int)
  atoms_1, atoms_2 = np.where(adjacency_matrix)
  for i in range(atoms_1.shape[0]):
      at_1 = atoms_1[i]
      at_2 = atoms_2[i]
      atom_graph[0, at_1, num_of_bonds_per_atom[at_1], 0] = 0
      atom_graph[0, at_1, num_of_bonds_per_atom[at_1], 1] = at_2
      bond_graph[0, at_1, num_of_bonds_per_atom[at_1], 0] = 0
      bond_graph[0, at_1, num_of_bonds_per_atom[at_1], 1] = at_1
      bond_graph[0, at_1, num_of_bonds_per_atom[at_1], 2] = at_2
      mask[0, at_1, num_of_bonds_per_atom[at_1]] = 1
      num_of_bonds_per_atom[at_1] += 1
  return atom_graph, bond_graph, mask


