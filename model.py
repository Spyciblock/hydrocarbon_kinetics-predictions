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
  input_atom = tf.keras.Input(shape=[num_atoms, num_atom_features])               # shape = num_atoms, num_atom_features
  input_pairs = tf.keras.Input(shape=[num_atoms, num_atoms, num_pairs_features])  # shape = num_atoms, num_atoms, num_pair_atoms_features
  atom_graph = tf.keras.Input(shape=[num_atoms, max_num_of_bonds, 2], dtype=tf.int32)  # shape = num_atoms, max_num_of_bonds, 2
  mask = tf.keras.Input(shape=[num_atoms, max_num_of_bonds, 1], dtype=tf.float32)      # shape = num_atoms, max_num_of_bonds, 1

  print("new_input_atom shape=",input_atom.shape)
  print("atom_graph shape=",atom_graph.shape)

  #Define number of hidden layers for every layers
  hidden_layers = 32        
  # First layer using input atoms
  atom_features=  tf.keras.layers.Dense(hidden_layers, activation='relu',name="Dense0")(input_atom) 

  # Get Embedding of features using a loop to update atoms features,evolve the system
  for i in range(depth):
    #For each atom, you store the features of the atoms bonded
    fatom_nei = tf.gather_nd(atom_features, atom_graph)  # shape [num_atoms,  max_number_of_bonds, num_atom_features]
    print("fatom_nei shape=",fatom_nei.shape)

    #Pass previous output into a layer of the NN model
    h_nei_atom = tf.keras.layers.Dense(hidden_layers, activation="relu",name='Dense1')(fatom_nei) # shape(batchsize,num_atoms,max_numbonds,hiddenlayers)
    print("h_nei_atom shape=",h_nei_atom.shape)
    # Add a mask to retrieve all significant outputs
    h_nei_atom = tf.math.multiply(h_nei_atom,tf.tile(mask, [1, 1, 1, h_nei_atom.shape[-1]]))   

    # Pass atoms features into a layer of the NN model
    f_self = tf.keras.layers.Dense(hidden_layers, activation="relu",name="Dense2")(atom_features) 
    print("f_self=",f_self.shape)
        
    # Combine previous output with the resulting features of the atoms bonded 
    f_nei = tf.reduce_sum(h_nei_atom, -2)         
    print("f_nei shape =",f_nei.shape)
    multiterms = f_nei*f_self
    print("multiterms shape =",multiterms.shape)

    #Add the previous output into a layer to get new atoms features
    final_step= tf.keras.layers.Dense(hidden_layers, activation="relu",name="Dense3")(multiterms) 
    print("final_step shape=",final_step.shape)

    #################################UPDATES atoms features ######################################
    # Use previous atoms and features of the atoms bonded to get new neighbor features
    nei_part = tf.keras.layers.Dense(hidden_layers, activation="relu",name="Dense4")(fatom_nei) 
    print("nei_part shape=",nei_part.shape)
    # Add mask to retrieve all significant outputs
    nei_part = tf.math.multiply(nei_part, mask)                     
    nei_partbis = tf.reduce_sum(nei_part ,-2)             
    print("nei_partbis shape=",nei_partbis.shape)
    #Combine atoms features and new neighbors
    new_part = tf.concat([atom_features, nei_partbis], 2)  
    print("new_part shape=",new_part.shape)

    # Pass previous output into a layer of the NN model and update atom features
    atom_features =tf.keras.layers.Dense(hidden_layers, activation='relu',name="Dense5")(new_part)    
    print("new atom_features shape=",atom_features.shape)

  final_input_atom=final_step                       
  print("final_input_atom shape=",final_input_atom.shape)

  # Prediction of reactivity score
  #Get matrices of shape[num_atoms,num_atoms, hiddenlayers] 
  atom_features_1 = tf.repeat(tf.reshape(final_input_atom, [-1, 1, num_atoms, hidden_layers]), [num_atoms], axis=1) 
  print("atom_features_1=",atom_features_1.shape)
  atom_features_2 = tf.repeat(tf.reshape(final_input_atom, [-1, num_atoms, 1, hidden_layers]), [num_atoms], axis=2) 
  print("atom_features_2=",atom_features_2.shape)
  
  #Add input pairs into a layer of the NN model
  print("input_pairs shape=",input_pairs.shape)
  new_input_pairs= tf.keras.layers.Dense(hidden_layers, activation="relu",name="Dense6")(input_pairs)   
  print("new_input_pairs shape=",new_input_pairs.shape)

  #Gather previous generated atoms features 
  concat = tf.concat([atom_features_1, atom_features_2], axis=-1)  
  midlayer = tf.keras.layers.Dense(hidden_layers, activation="relu",name="Dense7")(concat)  
  print("midlayer shape=",midlayer.shape)

  # Combine previous output with the new inputs_pairs then add it into the NN 
  concatenation_atom_and_pair_features = midlayer + new_input_pairs  
  print("concat shape=",concatenation_atom_and_pair_features.shape)
  endput= tf.keras.layers.Dense(hidden_layers, activation="relu",name="Dense8")(concatenation_atom_and_pair_features) 
  print("endput shape=",endput.shape) 

  #Retrieve the final binary value "reactions or not" using sigmoid activation function, convolutional layer
  preds = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(endput)
  print("final pred=",preds.shape)

  return tf.keras.Model(inputs=[input_atom, input_pairs, atom_graph, mask], outputs=preds)
