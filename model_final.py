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
def model_init(num_atoms, num_atom_features, num_pairs_features, max_num_of_bonds, hidden_size=32, depth=1, batchnorm=0, last_batchnorm=1, init='glorot_uniform', activ='relu'):
  # Inputs: atoms, pairs of atoms, graph of atoms, and mask to use
  input_atom = tf.keras.Input(shape=[num_atoms, num_atom_features])                    # shape = num_atoms, num_atom_features
  input_pairs = tf.keras.Input(shape=[num_atoms*(num_atoms - 1)//2, num_pairs_features])       # shape = num_atoms, num_atoms, num_pair_atoms_features
  atom_graph = tf.keras.Input(shape=[num_atoms, max_num_of_bonds, 2], dtype=tf.int32)  # shape = num_atoms, max_num_of_bonds, 2
  mask = tf.keras.Input(shape=[num_atoms, max_num_of_bonds, 1], dtype=tf.float32)      # shape = num_atoms, max_num_of_bonds, 1
  extract_pairs = tf.keras.Input(shape=[num_atoms*(num_atoms - 1)//2, 3], dtype=tf.int32)

  print("new_input_atom shape=",input_atom.shape)
  print("atom_graph shape=",atom_graph.shape)     
  
  # Use input atoms and encode it into our neural network
  atom_features = tf.keras.layers.Dense(4*hidden_size, activation=activ, kernel_initializer=init, name="Dense0-1")(input_atom) 
  atom_features = tf.keras.layers.Dense(4*hidden_size, activation=activ, kernel_initializer=init, name="Dense0-2")(atom_features) 
  atom_features = tf.keras.layers.Dense(hidden_size, activation=activ, kernel_initializer=init, name="Dense0-3")(atom_features) 
  if batchnorm == 1:
    atom_features = tf.keras.layers.BatchNormalization()(atom_features)

  dense_1 = tf.keras.layers.Dense(2*hidden_size, activation=activ, kernel_initializer=init, name='Dense1-1')
  dense_2 = tf.keras.layers.Dense(2*hidden_size, activation=activ, kernel_initializer=init, name="Dense1-2")
  dense_3 = tf.keras.layers.Dense(hidden_size, activation=None, kernel_initializer=init, name="Dense1-3")
  dense_4 = tf.keras.layers.Dense(2*hidden_size, activation=activ, kernel_initializer=init, name="Dense1-4")
  dense_5 = tf.keras.layers.Dense(2*hidden_size, activation=activ, kernel_initializer=init, name="Dense1-5")
  dense_6 = tf.keras.layers.Dense(hidden_size, activation=None, kernel_initializer=init, name='Dense1-6')
  dense_7 = tf.keras.layers.Dense(2*hidden_size, activation=activ, kernel_initializer=init, name="Dense1-7")
  dense_8 = tf.keras.layers.Dense(2*hidden_size, activation=activ, kernel_initializer=init, name="Dense1-8")
  dense_9 = tf.keras.layers.Dense(hidden_size, activation=None, kernel_initializer=init, name="Dense1-9")
  dense_10 = tf.keras.layers.Dense(2*hidden_size, activation=activ, kernel_initializer=init, name="Dense1-10")
  dense_11 = tf.keras.layers.Dense(2*hidden_size, activation=activ, kernel_initializer=init, name="Dense1-11")
  dense_12 = tf.keras.layers.Dense(2*hidden_size, activation=None, kernel_initializer=init, name="Dense1-12")
  dense_13 = tf.keras.layers.Dense(hidden_size, activation=activ, kernel_initializer=init, name="Dense1-13")
  dense_14 = tf.keras.layers.Dense(hidden_size, activation=activ, kernel_initializer=init, name="Dense1-14")

  # Get Embedding of features using a loop to update atoms features,evolve the system
  for i in range(depth):
    #For each atom, you store the features of the atoms bonded to form neighbors of atoms, the overall representation of the graph
    fatom_nei = tf.gather_nd(atom_features, atom_graph)  # shape [num_atoms,  max_number_of_bonds, num_atom_features]
    print("fatom_nei shape=",fatom_nei.shape)

    #Pass all generated neighbors into  the NN
    h_nei_atom = dense_1(fatom_nei)       # shape = num_atoms,max_numbonds,hiddensize)
    h_nei_atom = dense_2(h_nei_atom)
    h_nei_atom = dense_3(h_nei_atom)
    h_nei_atom = tf.keras.layers.Add()([fatom_nei, h_nei_atom])
    h_nei_atom = tf.keras.layers.ReLU()(h_nei_atom)
    if batchnorm == 1:
      h_nei_atom = tf.keras.layers.BatchNormalization()(h_nei_atom)
    print("h_nei_atom shape=",h_nei_atom.shape)
    # Add a mask to retrieve all existed bonds, because most of the atoms form less than the maximal number of bonds.
    h_nei_atom = tf.keras.layers.Multiply()([h_nei_atom, mask])   

    # Pass all atoms features into the NN model
    f_self = dense_4(atom_features)        #shape = num atoms, hiddensize
    f_self = dense_5(f_self)
    f_self = dense_6(f_self)
    f_self = tf.keras.layers.Add()([atom_features, f_self])
    f_self = tf.keras.layers.ReLU()(f_self)
    if batchnorm == 1:
      f_self = tf.keras.layers.BatchNormalization()(f_self)
    print("f_self=",f_self.shape)
        
    # Remove max num of bonds dimension and combine the neighbors features with the atom features to get all informations on atoms, a new representation
    f_nei = tf.reduce_sum(h_nei_atom, -2)   
    f_nei = tf.keras.layers.Reshape([num_atoms, -1])(h_nei_atom)
    f_nei = dense_13(f_nei)
    print(f_nei.shape)
    f_nei_2 = dense_7(f_nei)
    f_nei_2 = dense_8(f_nei_2)
    f_nei_2 = dense_9(f_nei_2)
    f_nei = tf.keras.layers.Add()([f_nei, f_nei_2])
    f_nei = tf.keras.layers.ReLU()(f_nei)
    print("f_nei shape =",f_nei.shape)
    multiterms = tf.keras.layers.Concatenate()([f_nei, f_self])                #shape = num_atoms,hidden_size
    print("multiterms shape =",multiterms.shape)

    #Add the new atoms representation into the neural network
    final_step= dense_10(multiterms)       #shape = num_atoms,hidden_size
    final_step= dense_11(final_step)
    final_step= dense_12(final_step)
    final_step = tf.keras.layers.Add()([multiterms, final_step])
    final_step = tf.keras.layers.ReLU()(final_step)
    final_step = dense_14(final_step)
    if batchnorm == 1:
      final_step = tf.keras.layers.BatchNormalization()(final_step)
    print("final_step shape=",final_step.shape)

    atom_features = final_step

    #################################UPDATES atoms features ######################################
    # Use the previous embedding of atoms features  and the features of the atoms bonded to update the neighbors of atoms
    # nei_part = dense_4(fatom_nei)         #shape = num_atoms,max_num_bonds,hidden_size
    # nei_part = dense_9(nei_part)
    # if batchnorm == 1:
    #   nei_part = tf.keras.layers.BatchNormalization()(nei_part)
    # print("nei_part shape=",nei_part.shape)
    # # Add mask to retrieve all existed bonds, because most of the atoms form less than the maximum possible number of bonds.
    # nei_part = tf.keras.layers.Multiply()([nei_part, mask])                     
    # nei_partbis = tf.reduce_sum(nei_part ,-2)                   #shape = num_atoms,hidden_size
    # print("nei_partbis shape=",nei_partbis.shape)

    # #Gather the orignal atoms features and the new pairs of atoms to form a new environment
    # new_part = tf.keras.layers.Concatenate(axis=2)([atom_features, nei_partbis])       #shape = num_atoms,hidden_size*2
    # print("new_part shape=",new_part.shape)

    # # Add  the new environment, composed of all new neighbors features and atom features into the NN , this updates the atom features reused at the beginning of this loop
    # atom_features = dense_5(new_part)      #shape = num_atoms,hidden_size
    # atom_features = dense_10(atom_features)
    # if batchnorm == 1:
    #   atom_features = tf.keras.layers.BatchNormalization()(atom_features)
    # print("new atom_features shape=",atom_features.shape)

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
  new_input_pairs= tf.keras.layers.Dense(hidden_size, activation=activ, kernel_initializer=init, name="Dense2-1")(input_pairs)   #shape = num_atoms,num_atoms,hidden_size
  new_input_pairs= tf.keras.layers.Dense(hidden_size, activation=activ, kernel_initializer=init, name="Dense2-2")(new_input_pairs)   #shape = num_atoms,num_atoms,hidden_size
  new_input_pairs= tf.keras.layers.Dense(hidden_size, activation=activ, kernel_initializer=init, name="Dense2-3")(new_input_pairs)   #shape = num_atoms,num_atoms,hidden_size
  new_input_pairs_2= new_input_pairs
  new_input_pairs= tf.keras.layers.Dense(hidden_size, activation=activ, kernel_initializer=init, name="Dense2-4")(new_input_pairs)   #shape = num_atoms,num_atoms,hidden_size
  new_input_pairs= tf.keras.layers.Dense(hidden_size, activation=activ, kernel_initializer=init, name="Dense2-5")(new_input_pairs)   #shape = num_atoms,num_atoms,hidden_size
  new_input_pairs= tf.keras.layers.Dense(hidden_size, activation=None, kernel_initializer=init, name="Dense2-6")(new_input_pairs)   #shape = num_atoms,num_atoms,hidden_size
  new_input_pairs = tf.keras.layers.Add()([new_input_pairs, new_input_pairs_2])
  new_input_pairs = tf.keras.layers.ReLU()(new_input_pairs)
  if batchnorm == 1:
    new_input_pairs = tf.keras.layers.BatchNormalization()(new_input_pairs)
  print("new_input_pairs shape=",new_input_pairs.shape)

  #Gather all informations of the generated atoms features 
  #concat = tf.keras.layers.Concatenate(axis=-1)([atom_features_1, atom_features_2])                             #shape = num_atoms,num_atoms,hidden_size
  concat = tf.keras.layers.Multiply()([atom_features_1, atom_features_2])
  print(concat.shape)
  concat = tf.gather_nd(concat, extract_pairs)
  print(concat.shape)
  midlayer = tf.keras.layers.Dense(hidden_size, activation=activ, kernel_initializer=init, name="Dense3-1")(concat)      #shape = num_atoms,num_atoms,hidden_size
  midlayer = tf.keras.layers.Dense(hidden_size, activation=activ, kernel_initializer=init, name="Dense3-2")(midlayer)      #shape = num_atoms,num_atoms,hidden_size
  midlayer = tf.keras.layers.Dense(hidden_size, activation=None, kernel_initializer=init, name="Dense3-3")(midlayer)      #shape = num_atoms,num_atoms,hidden_size
  midlayer = tf.keras.layers.Add()([concat, midlayer])
  midlayer = tf.keras.layers.ReLU()(midlayer)
  print(midlayer.shape)
  midlayer = tf.keras.layers.Dense(hidden_size, activation=activ, kernel_initializer=init, name="Dense3-4")(midlayer)      #shape = num_atoms,num_atoms,hidden_size
  midlayer = tf.keras.layers.Dense(hidden_size, activation=activ, kernel_initializer=init, name="Dense3-5")(midlayer)      #shape = num_atoms,num_atoms,hidden_size
  midlayer = tf.keras.layers.Dense(hidden_size, activation=None, kernel_initializer=init, name="Dense3-6")(midlayer)      #shape = num_atoms,num_atoms,hidden_size
  midlayer = tf.keras.layers.Add()([midlayer, concat])
  midlayer = tf.keras.layers.ReLU()(midlayer)
  if batchnorm == 1:
    midlayer = tf.keras.layers.BatchNormalization()(midlayer)
  print("midlayer shape=",midlayer.shape)

  # Combine previous atom features with the features of the pairs of atoms then add it into the NN
  # This will be used to get probabilities of reactions between all existed pairs of atoms
  concatenation_atom_and_pair_features = tf.keras.layers.Concatenate(axis=-1)([midlayer, new_input_pairs])
  print("concat shape=",concatenation_atom_and_pair_features.shape)
  endput= tf.keras.layers.Dense(hidden_size, activation=activ, kernel_initializer=init, name="Dense4-1")(concatenation_atom_and_pair_features)   #shape = num_atoms,num_atoms,hidden_size
  endput= tf.keras.layers.Dense(hidden_size, activation=activ, kernel_initializer=init, name="Dense4-2")(endput)   #shape = num_atoms,num_atoms,hidden_size
  endput= tf.keras.layers.Dense(2*hidden_size, activation=None, kernel_initializer=init, name="Dense4-3")(endput)   #shape = num_atoms,num_atoms,hidden_size
  endput= tf.keras.layers.Add()([endput, concatenation_atom_and_pair_features])
  endput= tf.keras.layers.ReLU()(endput)
  endput= tf.keras.layers.Dense(hidden_size, activation=activ, kernel_initializer=init, name="Dense4-4")(endput)   #shape = num_atoms,num_atoms,hidden_size
  endput= tf.keras.layers.Dense(hidden_size, activation=activ, kernel_initializer=init, name="Dense4-5")(endput)   #shape = num_atoms,num_atoms,hidden_size
  endput= tf.keras.layers.Dense(hidden_size, activation=activ, kernel_initializer=init, name="Dense4-6")(endput)   #shape = num_atoms,num_atoms,hidden_size
  if batchnorm == 1:
    endput = tf.keras.layers.BatchNormalization()(endput)

  print("endput shape=",endput.shape) 

  #Retrieve the final probability between 0 and 1, it shows if a reaction will occur or not using sigmoid activation function, convolutional layer
  #preds = tf.keras.layers.Conv2D(1, 1, activation=None)(endput)      #shape = num_atoms,num_atoms,1
  preds = tf.keras.layers.Dense(1, activation=None, kernel_initializer=init, kernel_regularizer="l2", name="Dense_last")(endput)
  print(preds.shape)
  preds = tf.reduce_sum(preds, axis=-1)
  if last_batchnorm == 1:
    preds = tf.keras.layers.BatchNormalization()(preds)
  print(preds.shape)
  preds_probs = tf.keras.layers.Softmax(name='prediction_probs')(preds)
  preds_time = tf.keras.layers.Flatten()(preds)
  preds_time = tf.math.exp(preds_time)
  preds_time = tf.reduce_sum(preds_time, axis=-1, keepdims=True)
  preds_time = tf.ones_like(preds_time)
  preds_time = tf.keras.layers.Lambda(lambda x: x, name='prediction_time')(preds_time)
  # preds_time = tf.keras.layers.Dense(1, kernel_initializer='ones', trainable=False,kernel_regularizer="l2", name='prediction_time')(preds_time)
  # preds_time = tf.expand_dims(preds_time, axis=-1, name='prediction_time')
  print("final pred=",preds.shape)
  print("Pred_probs = ", preds_probs.shape)
  print("Preds_time = ", preds_time.shape)

  return tf.keras.Model(inputs=[input_atom, input_pairs, atom_graph, mask, extract_pairs], outputs=[preds_probs, preds_time])
