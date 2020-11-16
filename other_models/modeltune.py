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
def model_init(num_atoms, num_atom_features, num_pairs_features, max_num_of_bonds, depth,lr,hidden_size,nbdense):
  # Inputs: atoms, pairs of atoms, graph of atoms, and mask to use
  input_atom = tf.keras.Input(shape=[num_atoms, num_atom_features])                    # shape = num_atoms, num_atom_features
  input_pairs = tf.keras.Input(shape=[num_atoms*(num_atoms - 1)//2, num_pairs_features])       # shape = num_atoms, num_atoms, num_pair_atoms_features
  atom_graph = tf.keras.Input(shape=[num_atoms, max_num_of_bonds, 2], dtype=tf.int32)  # shape = num_atoms, max_num_of_bonds, 2
  mask = tf.keras.Input(shape=[num_atoms, max_num_of_bonds, 1], dtype=tf.float32)      # shape = num_atoms, max_num_of_bonds, 1
  extract_pairs = tf.keras.Input(shape=[num_atoms*(num_atoms - 1)//2, 3], dtype=tf.int32)

  print("new_input_atom shape=",input_atom.shape)
  print("atom_graph shape=",atom_graph.shape)

  #Define size of a layer
  #hidden_size = 32        
  
  # Use input atoms and encode it into our neural network
  atom_features = tf.keras.layers.Dense(hidden_size, activation='relu',kernel_regularizer="l1_l2",name="Dense0")(input_atom)
  dense_1 = tf.keras.layers.Dense(hidden_size, activation="relu",kernel_regularizer="l1_l2",name='Dense1')
  dense_1b = tf.keras.layers.Dense(hidden_size, activation="relu",kernel_regularizer="l1_l2",name='Dense1b')
  dense_1c = tf.keras.layers.Dense(hidden_size, activation="relu",kernel_regularizer="l1_l2",name='Dense1c')

  dense_2 = tf.keras.layers.Dense(hidden_size, activation="relu",kernel_regularizer="l1_l2",name="Dense2")
  dense_2b = tf.keras.layers.Dense(hidden_size, activation="relu",kernel_regularizer="l1_l2",name="Dense2b")
  dense_2c = tf.keras.layers.Dense(hidden_size, activation="relu",kernel_regularizer="l1_l2",name="Dense2c")

  dense_3 = tf.keras.layers.Dense(hidden_size, activation="relu",kernel_regularizer="l1_l2",name="Dense3")
  dense_3b = tf.keras.layers.Dense(hidden_size, activation="relu",kernel_regularizer="l1_l2",name="Dense3b")
  dense_3c = tf.keras.layers.Dense(hidden_size, activation="relu",kernel_regularizer="l1_l2",name="Dense3c")

  dense_4 = tf.keras.layers.Dense(hidden_size, activation="relu",kernel_regularizer="l1_l2",name="Dense4")
  dense_4b = tf.keras.layers.Dense(hidden_size, activation="relu",kernel_regularizer="l1_l2",name="Dense4b")
  dense_4c = tf.keras.layers.Dense(hidden_size, activation="relu",kernel_regularizer="l1_l2",name="Dense4c")

  dense_5 = tf.keras.layers.Dense(hidden_size, activation='relu',kernel_regularizer="l1_l2",name="Dense5")
  dense_5b = tf.keras.layers.Dense(hidden_size, activation="relu",kernel_regularizer="l1_l2",name="Dense5b")
  dense_5c = tf.keras.layers.Dense(hidden_size, activation="relu",kernel_regularizer="l1_l2",name="Dense5c")
  # Get Embedding of features using a loop to update atoms features,evolve the system
  for i in range(depth):
    #For each atom, you store the features of the atoms bonded to form neighbors of atoms, the overall representation of the graph
    fatom_nei = tf.gather_nd(atom_features, atom_graph)  # shape [num_atoms,  max_number_of_bonds, num_atom_features]
    print("fatom_nei shape=",fatom_nei.shape)

    #Pass all generated neighbors into  the NN
    h_nei_atom = dense_1(fatom_nei)       # shape = num_atoms,max_numbonds,hiddensize)
    if nbdense>1:
      h_nei_atom = dense_1b(h_nei_atom)
      h_nei_atom = dense_1c(h_nei_atom)
    print("h_nei_atom shape=",h_nei_atom.shape)
    # Add a mask to retrieve all existed bonds, because most of the atoms form less than the maximal number of bonds.
    h_nei_atom = tf.keras.layers.Multiply()([h_nei_atom, mask])   

    # Pass all atoms features into the NN model
    f_self = dense_2(atom_features)        #shape = num atoms, hiddensize
    if nbdense>1:
      f_self = dense_2b(f_self)
      f_self = dense_2c(f_self)

    print("f_self=",f_self.shape)
        
    # Remove max num of bonds dimension and combine the neighbors features with the atom features to get all informations on atoms, a new representation
    f_nei = tf.reduce_sum(h_nei_atom, -2)         
    print("f_nei shape =",f_nei.shape)
    multiterms = tf.keras.layers.Multiply()([f_nei, f_self])                #shape = num_atoms,hidden_size
    print("multiterms shape =",multiterms.shape)

    #Add the new atoms representation into the neural network
    final_step= dense_3(multiterms)       #shape = num_atoms,hidden_size
    if nbdense>1:
      final_step= dense_3b(final_step) 
      final_step= dense_3c(final_step) 
    print("final_step shape=",final_step.shape)

    #################################UPDATES atoms features ######################################
    # Use the previous embedding of atoms features  and the features of the atoms bonded to update the neighbors of atoms
    nei_part = dense_4(fatom_nei)         #shape = num_atoms,max_num_bonds,hidden_size
    if nbdense>1:
      nei_part = dense_4b(fatom_nei)  
      nei_part = dense_4c(nei_part)   
    print("nei_part shape=",nei_part.shape)
    # Add mask to retrieve all existed bonds, because most of the atoms form less than the maximum possible number of bonds.
    nei_part = tf.keras.layers.Multiply()([nei_part, mask])                     
    nei_partbis = tf.reduce_sum(nei_part ,-2)                   #shape = num_atoms,hidden_size
    print("nei_partbis shape=",nei_partbis.shape)

    #Gather the orignal atoms features and the new pairs of atoms to form a new environment
    new_part = tf.keras.layers.Concatenate(axis=2)([atom_features, nei_partbis])       #shape = num_atoms,hidden_size*2
    print("new_part shape=",new_part.shape)

    # Add  the new environment, composed of all new neighbors features and atom features into the NN , this updates the atom features reused at the beginning of this loop
    atom_features = dense_5(new_part)      #shape = num_atoms,hidden_size
    if nbdense>1:
      atom_features = dense_5b(atom_features) 
      atom_features = dense_5c(atom_features) 
    print("new atom_features shape=",atom_features.shape)

  final_input_atom=final_step                       
  print("final_input_atom shape=",final_input_atom.shape)

  # Prediction of reactivity score
  # Get matrices of shape[num_atoms,num_atoms, hiddenlayers] , this regroups all specific informations of atoms (atoms type, cycle size, molecule size)for each dimension
  atom_features_1 = tf.keras.layers.Reshape((1, num_atoms, hidden_size))(final_input_atom)
  print("atom_features_1=",atom_features_1.shape)
  atom_features_1 = tf.repeat(atom_features_1, [num_atoms], axis=1) 
  print("atom_features_1=",atom_features_1.shape)
  atom_features_2 = tf.keras.layers.Reshape((num_atoms, 1, hidden_size))(final_input_atom)
  atom_features_2 = tf.repeat(atom_features_2, [num_atoms], axis=2) 
  print("atom_features_2=",atom_features_2.shape)

  #Add informations of pair of atoms,features(bonded, same cycle, same molecule) into the NN 
  # new_input_pairs = tf.gather_nd(input_pairs, extract_pairs)
  print("input_pairs shape=",input_pairs.shape)
  new_input_pairs= tf.keras.layers.Dense(hidden_size, activation="relu",kernel_regularizer="l1_l2",name="Dense6")(input_pairs)   #shape = num_atoms,num_atoms,hidden_size
  if nbdense>1:
    new_input_pairs=tf.keras.layers.Dense(hidden_size, activation="relu",kernel_regularizer="l1_l2",name="Dense6b")(new_input_pairs)
    new_input_pairs=tf.keras.layers.Dense(hidden_size, activation="relu",kernel_regularizer="l1_l2",name="Dense6c")(new_input_pairs)
    
  print("new_input_pairs shape=",new_input_pairs.shape)

  #Gather all informations of the generated atoms features 
  concat = tf.keras.layers.Concatenate(axis=-1)([atom_features_1, atom_features_2])                             #shape = num_atoms,num_atoms,hidden_size
  concat = tf.gather_nd(concat, extract_pairs)
  midlayer0 = tf.keras.layers.Dense(hidden_size, activation=None,kernel_regularizer="l1_l2",name="Dense7")(concat)      #shape = num_atoms,num_atoms,hidden_size
  print("midlayer shape=",midlayer0.shape)
  layer= tf.keras.layers.LeakyReLU(alpha=0.3)
  midlayer = layer(midlayer0)

  # Combine previous atom features with the features of the pairs of atoms then add it into the NN
  # This will be used to get probabilities of reactions between all existed pairs of atoms
  concatenation_atom_and_pair_features = tf.keras.layers.Concatenate(axis=-1)([midlayer, new_input_pairs])
  print("concat shape=",concatenation_atom_and_pair_features.shape)
  endput0= tf.keras.layers.Dense(hidden_size, activation=None,kernel_regularizer="l1_l2",name="Dense8")(concatenation_atom_and_pair_features)   #shape = num_atoms,num_atoms,hidden_size
  print("endput shape=",endput0.shape)
  layer2= tf.keras.layers.LeakyReLU(alpha=0.3)
  endput = layer2(endput0)

  #Retrieve the final probability between 0 and 1, it shows if a reaction will occur or not using sigmoid activation function, convolutional layer

  preds = tf.reduce_sum(endput, axis=-1)
  print(preds.shape)
  preds_probs = tf.keras.layers.Softmax(name='prediction_probs')(preds)
  preds_time = tf.keras.layers.Flatten()(preds)
  
  preds_time = tf.math.exp(preds_time)
  preds_time = tf.reduce_sum(preds_time, axis=-1, keepdims=True)
  preds_time = tf.keras.layers.Lambda(lambda x: x, name='prediction_time')(preds_time)
  print("final pred=",preds.shape)
  print("Pred_probs = ", preds_probs.shape)
  print("Preds_time = ", preds_time.shape)

  print("learning rate used=",lr)
  print("hiddensize used=",hidden_size)
  print("depth used=",depth)
  return tf.keras.Model(inputs=[input_atom, input_pairs, atom_graph, mask, extract_pairs], outputs=[preds_probs, preds_time])
