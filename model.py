import numpy as np
import pickle
from scipy import sparse
import tensorflow as tf

from data_generator import DataLoader
from kinetic_monte_carlo import run_KMC
from utils import *
from molecules import MoleculeList
import os,datetime  
import boto3

# Model Architecture v1 initialization
def model_init(num_atoms, num_atom_features, num_pairs_features, max_num_of_bonds, depth=1):
  # Inputs: atoms, pairs of atoms, graph of atoms, and mask to use
  input_atom = tf.keras.Input(shape=[num_atoms, num_atom_features])                    # shape = num_atoms, num_atom_features
  input_pairs = tf.keras.Input(shape=[num_atoms, num_atoms, num_pairs_features])       # shape = num_atoms, num_atoms, num_pair_atoms_features
  atom_graph = tf.keras.Input(shape=[num_atoms, max_num_of_bonds, 2], dtype=tf.int32)  # shape = num_atoms, max_num_of_bonds, 2
  mask = tf.keras.Input(shape=[num_atoms, max_num_of_bonds, 1], dtype=tf.float32)      # shape = num_atoms, max_num_of_bonds, 1

  print("new_input_atom shape=",input_atom.shape)
  print("atom_graph shape=",atom_graph.shape)

  #Define size of a layer
  hidden_size = 32        
  
  # Use input atoms and encode it into our neural network
  atom_features=  tf.keras.layers.Dense(hidden_size, activation='relu',name="Dense0")(input_atom) 

  # Get Embedding of features using a loop to update atoms features,evolve the system
  for i in range(depth):
    #For each atom, you store the features of the atoms bonded to form neighbors of atoms, the overall representation of the graph
    fatom_nei = tf.gather_nd(atom_features, atom_graph)  # shape [num_atoms,  max_number_of_bonds, num_atom_features]
    print("fatom_nei shape=",fatom_nei.shape)

    #Pass all generated neighbors into  the NN
    h_nei_atom = tf.keras.layers.Dense(hidden_size, activation="relu",name='Dense1')(fatom_nei)       # shape = num_atoms,max_numbonds,hiddensize)
    print("h_nei_atom shape=",h_nei_atom.shape)
    # Add a mask to retrieve all existed bonds, because most of the atoms form less than the maximal number of bonds.
    h_nei_atom = tf.math.multiply(h_nei_atom,tf.tile(mask, [1, 1, 1, h_nei_atom.shape[-1]]))   

    # Pass all atoms features into the NN model
    f_self = tf.keras.layers.Dense(hidden_size, activation="relu",name="Dense2")(atom_features)        #shape = num atoms, hiddensize
    print("f_self=",f_self.shape)
        
    # Remove max num of bonds dimension and combine the neighbors features with the atom features to get all informations on atoms, a new representation
    f_nei = tf.reduce_sum(h_nei_atom, -2)         
    print("f_nei shape =",f_nei.shape)
    multiterms = f_nei*f_self                   #shape = num_atoms,hidden_size
    print("multiterms shape =",multiterms.shape)

    #Add the new atoms representation into the neural network
    final_step= tf.keras.layers.Dense(hidden_size, activation="relu",name="Dense3")(multiterms)       #shape = num_atoms,hidden_size
    print("final_step shape=",final_step.shape)

    #################################UPDATES atoms features ######################################
    # Use the previous embedding of atoms features  and the features of the atoms bonded to update the neighbors of atoms
    nei_part = tf.keras.layers.Dense(hidden_size, activation="relu",name="Dense4")(fatom_nei)         #shape = num_atoms,max_num_bonds,hidden_size
    print("nei_part shape=",nei_part.shape)
    # Add mask to retrieve all existed bonds, because most of the atoms form less than the maximum possible number of bonds.
    nei_part = tf.math.multiply(nei_part, mask)                     
    nei_partbis = tf.reduce_sum(nei_part ,-2)                   #shape = num_atoms,hidden_size
    print("nei_partbis shape=",nei_partbis.shape)

    #Gather the orignal atoms features and the new pairs of atoms to form a new environment
    new_part = tf.concat([atom_features, nei_partbis], 2)       #shape = num_atoms,hidden_size*2
    print("new_part shape=",new_part.shape)

    # Add  the new environment, composed of all new neighbors features and atom features into the NN , this updates the atom features reused at the beginning of this loop
    atom_features =tf.keras.layers.Dense(hidden_size, activation='relu',name="Dense5")(new_part)      #shape = num_atoms,hidden_size
    print("new atom_features shape=",atom_features.shape)

  final_input_atom=final_step                       
  print("final_input_atom shape=",final_input_atom.shape)

  # Prediction of reactivity score
  #Get matrices of shape[num_atoms,num_atoms, hiddenlayers] , this regroups all specific informations of atoms (atoms type, cycle size, molecule size)for each dimension
  atom_features_1 = tf.repeat(tf.reshape(final_input_atom, [-1, 1, num_atoms, hidden_size]), [num_atoms], axis=1) 
  print("atom_features_1=",atom_features_1.shape)
  atom_features_2 = tf.repeat(tf.reshape(final_input_atom, [-1, num_atoms, 1, hidden_size]), [num_atoms], axis=2) 
  print("atom_features_2=",atom_features_2.shape)
  
  #Add informations of pair of atoms,features(bonded, same cycle, same molecule) into the NN 
  print("input_pairs shape=",input_pairs.shape)
  new_input_pairs= tf.keras.layers.Dense(hidden_size, activation="relu",name="Dense6")(input_pairs)   #shape = num_atoms,num_atoms,hidden_size
  print("new_input_pairs shape=",new_input_pairs.shape)

  #Gather all informations of the generated atoms features 
  concat = tf.concat([atom_features_1, atom_features_2], axis=-1)                             #shape = num_atoms,num_atoms,hidden_size
  midlayer = tf.keras.layers.Dense(hidden_size, activation="relu",name="Dense7")(concat)      #shape = num_atoms,num_atoms,hidden_size
  print("midlayer shape=",midlayer.shape)

  # Combine previous atom features with the features of the pairs of atoms then add it into the NN
  # This will be used to get probabilities of reactions between all existed pairs of atoms
  concatenation_atom_and_pair_features = midlayer + new_input_pairs  
  print("concat shape=",concatenation_atom_and_pair_features.shape)
  endput= tf.keras.layers.Dense(hidden_size, activation="relu",name="Dense8")(concatenation_atom_and_pair_features)   #shape = num_atoms,num_atoms,hidden_size
  print("endput shape=",endput.shape) 

  #Retrieve the final probability between 0 and 1, it shows if a reaction will occur or not using sigmoid activation function, convolutional layer
  preds = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(endput)      #shape = num_atoms,num_atoms,1
  print("final pred=",preds.shape)

  return tf.keras.Model(inputs=[input_atom, input_pairs, atom_graph, mask], outputs=preds)
