import os
FOLDERNAME = os.getcwd() + '/'

import boto3
from pathlib import Path
import argparse
import json
import numpy as np
import pickle
import pdb
from scipy import sparse
from shutil import copyfile
import tensorflow as tf
import datetime  

from data_generator import DataLoader
from data_generator_bucket import DataLoaderBucket
from kinetic_monte_carlo import run_KMC
from utils import *
from molecules import MoleculeList
from model_final import model_init

# Used to plot loss and accuracy evolution
def plot_hist(foldername_save, history,imgname):
  
  if not os.path.exists(foldername_save + "plots/"+imgname):
    os.mkdir(foldername_save + "plots/"+imgname)
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  
  prediction_probs_loss = history.history['prediction_probs_loss']
  val_prediction_probs_loss = history.history['val_prediction_probs_loss']

  prediction_time_loss = history.history['prediction_time_loss']
  val_prediction_time_loss = history.history['val_prediction_time_loss']

  plt.figure(1)
  plt.plot(loss, label='Training Loss')
  plt.plot(val_loss, label='Validation Loss')
  plt.ylabel('Loss')
  plt.autoscale(enable=True, axis='both', tight=None)
  plt.title('Training and Validation Loss')
  plt.xlabel('epoch')
  plt.savefig(foldername_save + "plots/"+imgname+"/loss.png")
  plt.close()
 
  plt.figure(2)
  plt.plot(prediction_probs_loss, label='Training prediction probs loss')
  plt.plot(val_prediction_probs_loss, label='Validation prediction probs loss')
  plt.ylabel('Loss')
  plt.autoscale(enable=True, axis='both', tight=None)
  plt.title('Training and Validation prediction probs loss')
  plt.savefig(foldername_save + "plots/"+imgname+"/probsloss.png")
  plt.close()

  plt.figure(3)
  plt.plot(prediction_time_loss, label='Training prediction time loss')
  plt.plot(val_prediction_time_loss, label='Validation prediction time loss')
  plt.ylabel('Loss')
  plt.autoscale(enable=True, axis='both', tight=None)
  plt.title('Training and Validation prediction time loss')
  plt.savefig(foldername_save + "plots/"+imgname+"/timeloss.png")
  plt.close()

def scheduler(epoch, lr):
    if epoch <= 9:
        lr = 0.001
    elif epoch <= 13:
        lr = 0.0001
    elif epoch <= 15:
        lr = 0.01
    elif epoch <= 24:
        lr = 0.001
    else:
        lr = 0.0001
    return lr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('param_file', type=Path)
    args = parser.parse_args()
    param = json.load(args.param_file.open())


    use_bucket = param['use_bucket']

    #Select the way to access data: from bucket or disk storage
    if use_bucket == 0:		
        foldername = param['foldername']

    else:
        foldername = param['foldername']
    # Define which features to use with boolean value
    atom_features_bool = {"Atom types": param['atom_features_atom_types'],
        "Cycle size": param['atom_features_cycle_size'],
        "Molecule size": param['atom_features_molecule_size'],
        "Number of neighbors": param['atom_features_num_of_neighbors']}
    pairs_features_bool = {"Bonded": param['pairs_features_bonded'],
            "Same cycle": param['pairs_features_same_cycle'],
            "Same molecule": param['pairs_features_same_molecule']}
    max_num_of_bonds = param['max_num_of_bonds']

    # Get useful constants used for the model
    num_atoms, num_atom_features, num_pairs_features, num_timesteps = get_constants(foldername, atom_features_bool, pairs_features_bool)

    print("Number of atoms: " + str(num_atoms))
    print("Number of time steps: " + str(num_timesteps))
    print("Max number of bonds an atom can form: " + str(max_num_of_bonds))
    print("Number of atom features: " + str(num_atom_features))
    print("Number of pairs features: " + str(num_pairs_features))

    # Normalization constants
    molecule_size_normalizer = param['molecule_size_normalizer']
    cycle_size_normalizer = param['cycle_size_normalizer']
    validation_percentage = param['validation_percentage']

    # Define how you load your training data
    use_generator = param['use_generator']        # 1 if you want to use a generator, 0 if you want to load all your values at the beginning
    first_examples_bool = param['first_examples_bool']   # 1 to load the frames 0 to num_examples, 0 to load random num_examples in the num_timesteps frames
                              # Only useful if use_generator == 0
    # Define the validation dataset
    num_examples_val = param['num_examples_val']         # Same as num_examples, must be < validation_percentage * num_timesteps
    first_examples_val_bool = param['first_examples_val_bool']  # Same as first_examples_bool

    # Define training parameters
    batch_size = param['batch_size']            # Not too high 
    num_epochs = param['num_epochs']
    num_examples_per_epoch = param['num_examples_per_epoch']      # If use_generator == 0, number of examples that are loaded, else this helps define steps_per_epoch in the training
    num_examples_per_epoch = (num_examples_per_epoch//batch_size)*batch_size
    # Cannot be > 200 if use_generator == 0, RAM will crash
    hidden_size = param['hidden_size']
    depth = param['depth']
    batchnorm = param['batchnorm']
    last_batchnorm = param['last_batchnorm']
    activ = param['activ']
    initializer= param['initializer']
    second_part_loop = param['second_part_loop']
    add_layers_1 = param['add_layers_1']
    reduce_sum_bool = param['reduce_sum_bool']
    loop_multiplier = param['loop_multiplier']
    pairs_encoding_multiplier = param['pairs_encoding_multiplier']

    learning_rate = param['learning_rate']
    learning_rate_scheduler_bool = param['learning_rate_scheduler_bool']
    myLosses = {'prediction_probs': param['loss_probs'], 'prediction_time': param['loss_time']}
    myLossesWeights = {'prediction_probs': 10**param['loss_weight_probs'], 'prediction_time': 0}

    load_model_bool = param['load_model_bool']
    model_to_load = param['model_to_load']

    name_training = param['name_training']
    foldername_save = param['foldername_save']
    if not os.path.exists(foldername_save + "plots/" + name_training + "/"):
      os.mkdir(foldername_save + "plots/" + name_training + "/")
    if not os.path.exists(foldername_save + "saved_model/" + name_training + "/"):
      os.mkdir(foldername_save + "saved_model/" + name_training + "/")
 
    #Select how to load and how to preprocess data according to the path selected,using data generator or not that fits with our computational ressources
    # for training and validation data
    if use_bucket == 0:
        data_loader = DataLoader(foldername, atom_features_bool, pairs_features_bool,
            num_atom_features, num_pairs_features, num_atoms, 
            molecule_size_normalizer, cycle_size_normalizer, max_num_of_bonds,
            num_timesteps, validation_percentage)
    else:
        data_loader = DataLoaderBucket(foldername, atom_features_bool, pairs_features_bool,
            num_atom_features, num_pairs_features, num_atoms, 
            molecule_size_normalizer, cycle_size_normalizer, max_num_of_bonds,
            num_timesteps, validation_percentage)

    if use_generator:
        data = tf.data.Dataset.from_generator(data_loader.get_data_with_generator, args=[num_examples_per_epoch, batch_size], output_types=((tf.float32, tf.float32, tf.int32, tf.float32, tf.int32), (tf.float32, tf.float32)), output_shapes = (((num_atoms, num_atom_features), (num_atoms*(num_atoms - 1)//2, num_pairs_features), (num_atoms, max_num_of_bonds, 2), (num_atoms, max_num_of_bonds, 1), (num_atoms*(num_atoms - 1)//2, 3)), ((num_atoms*(num_atoms - 1)//2), 1)))
        print(data)
    else:
        X_input_atom, X_input_pairs, X_atom_graph, X_mask, X_extract_pairs, Y, Y_time = data_loader.get_data_no_generator(num_examples_per_epoch, first_examples_bool, 'train')
        print("X_input_atom shape:" + str(X_input_atom.shape))
        print("X_input_pairs shape: " + str(X_input_pairs.shape))
        print("X_atom_graph shape: " + str(X_atom_graph.shape))
        print("X_mask shape: " + str(X_mask.shape))
        print("X_extract_pairs shape:" + str(X_extract_pairs.shape))
        print("Y shape: " + str(Y.shape))
        print("Y_time shape: " + str(Y_time.shape))
    X_input_atom_val, X_input_pairs_val, X_atom_graph_val, X_mask_val, X_extract_pairs_val, Y_val, Y_time_val = data_loader.get_data_no_generator(num_examples_val, first_examples_val_bool, 'val')
    print("")
    print("X_input_atom_val shape:" + str(X_input_atom_val.shape))
    print("X_input_pairs_val shape: " + str(X_input_pairs_val.shape))
    print("X_atom_graph_val shape: " + str(X_atom_graph_val.shape))
    print("X_mask_val shape: " + str(X_mask_val.shape))
    print("X_extract_pairs_val shape:" + str(X_extract_pairs_val.shape))
    print("Y_val shape: " + str(Y_val.shape))
    print("Y_time_val shape: " + str(Y_time_val.shape))

    # Initialize model with inputs, and additionnal methods for training
    if load_model_bool == 0:
        #model = model_init(num_atoms, num_atom_features, num_pairs_features, max_num_of_bonds, hidden_size=hidden_size, depth=depth, batchnorm=batchnorm, last_batchnorm=last_batchnorm, activ=activ, init=initalizer, second_part_loop=second_part_loop, add_layers_1=add_layers_1, reduce_sum_bool=reduce_sum_bool, loop_multiplier=loop_multiplier, pairs_encoding_multiplier=pairs_encoding_multiplier)
        model = model_init_2(num_atoms, num_atom_features, num_pairs_features, max_num_of_bonds, hidden_size=hidden_size, depth=depth, batchnorm=batchnorm, last_batchnorm=last_batchnorm, init=initializer, activ=activ)
        #model = model_init_3(num_atoms, num_atom_features, num_pairs_features, max_num_of_bonds, hidden_size=hidden_size, depth=depth, batchnorm=batchnorm, last_batchnorm=last_batchnorm, init=initializer, activ=activ)
    else:
        model = tf.keras.models.load_model('./saved_model/' + model_to_load)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, mode='min', restore_best_weights=True)
    callbacks_list = [early_stopping]
    if learning_rate_scheduler_bool == 1:
        callbacks_list.append(tf.keras.callbacks.LearningRateScheduler(scheduler))
    lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.3, min_delta=0.1, patience=2)
    callbacks_list.append(lr_on_plateau)
    #lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
    #callbacks_list = [lr_scheduler]
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(foldername_save + 'saved_model/' + str(name_training) + "/", monitor='loss', save_best_only=True)
    callbacks_list.append(model_checkpoint)
    

    # Build the model by using the adam optimizer and mse loss, accuracy metric 
    adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam,
            loss=myLosses, loss_weights=myLossesWeights, metrics=['accuracy'])

    # Fit the model with the appropriated  data pipeline method,so the full dataset does not need to fit into memory.
    if use_generator == 1:
        history = model.fit(data.repeat().batch(batch_size=batch_size).prefetch(tf.data.experimental.AUTOTUNE), epochs=num_epochs, steps_per_epoch=int(num_examples_per_epoch/batch_size), 
            validation_data=([X_input_atom_val, X_input_pairs_val, X_atom_graph_val, X_mask_val, X_extract_pairs_val], [Y_val, Y_time_val]), callbacks=callbacks_list) 
    else:
        history = model.fit([X_input_atom, X_input_pairs, X_atom_graph, X_mask, X_extract_pairs], [Y, Y_time], 
            batch_size=batch_size, epochs=num_epochs, 
            validation_data=([X_input_atom_val, X_input_pairs_val, X_atom_graph_val, X_mask_val, X_extract_pairs_val], [Y_val, Y_time_val]), callbacks=callbacks_list)

    print(model.summary())
    #Save the model
    model.save(foldername_save + 'saved_model/' + str(name_training) + "/") 
    #copyfile('./model_2.py', foldername_save + 'saved_model/' + str(name_training) + "/")

    #Save  loss history into png file
    plot_hist(foldername_save, history,name_training)

    # Run the KMC with the ML model
    num_iterations = param['num_iterations_KMC']
    bond_change, first_frame, time_range = run_KMC(model, num_iterations, foldername, atom_features_bool, pairs_features_bool,
                    num_atom_features, num_pairs_features, num_atoms, 
                    molecule_size_normalizer, cycle_size_normalizer, max_num_of_bonds,
                    num_timesteps, validation_percentage)

    # Get atom types and num_of_types
    atom_types = np.load(foldername + 'input_atom/atom_types.npy')
    num_of_types = np.unique(atom_types).shape[0]

    # Get the evolution of molecules with the ML model
    molecules = MoleculeList(num_atoms, atom_types, num_of_types)
    molecules_per_frame, molecule_list = molecules.get_molecule_full(first_frame, bond_change)

    # Get the evolution of molecules for the ground truth
    molecules_MD = MoleculeList(num_atoms, atom_types, num_of_types)
    first_frame_MD = np.load(foldername + 'first_frame.npy')
    bond_change_MD = pickle.load(open(foldername + "bond_change.pkl", "rb"))
    bond_change_MD, time_range_MD = keep_only_dictionary_with_iterations(bond_change_MD, num_iterations)
    molecules_per_frame_MD, molecule_list_MD = molecules_MD.get_molecule_full(first_frame_MD, bond_change_MD)

    # Plot different molecules of interest
    molecules_of_interest = np.array([[4, 10, 3, 10, 0], [1, 4, 0, 4, 0]])
    plot_molecules_of_interest(foldername_save, name_training, molecules_per_frame, molecule_list, time_range, molecules_per_frame_MD, molecule_list_MD, time_range_MD*0.012, molecules_of_interest) 

    #Save  loss history into png file
    #plot_hist(history,"09112020")
