import boto3
import numpy as np
import pickle
import pdb
from scipy import sparse
import tensorflow as tf
import os,datetime  

from data_generator import DataLoader
from data_generator_bucket import DataLoaderBucket
from kinetic_monte_carlo import run_KMC
from utils import *
from molecules import MoleculeList
from model import model_init

# Used to plot loss and accuracy evolution
def plot_hist(history,imgname):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.ylim([-1,max(plt.ylim())])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig("./plots/"+imgname+".png")
    plt.close() 




if __name__ == "__main__":
    use_bucket = 0

    #Select the way to access data: from bucket or disk storage
    if use_bucket == 0:		
        foldername ='/home/ubuntu/Data/C4H10_600ps/'
    else:
        foldername = 'C4H10_10ps/'
    # Define which features to use
    atom_features_bool = {"Atom types": 1,
        "Cycle size": 1,
        "Molecule size": 1}
    pairs_features_bool = {"Bonded": 1,
            "Same cycle": 1,
            "Same molecule": 1}
    max_num_of_bonds = 10

    # Get useful constants used for the model
    num_atoms, num_atom_features, num_pairs_features, num_timesteps = get_constants(foldername, atom_features_bool, pairs_features_bool)

    print("Number of atoms: " + str(num_atoms))
    print("Number of time steps: " + str(num_timesteps))
    print("Max number of bonds an atom can form: " + str(max_num_of_bonds))
    print("Number of atom features: " + str(num_atom_features))
    print("Number of pairs features: " + str(num_pairs_features))

    # Normalization constants
    molecule_size_normalizer = 200
    cycle_size_normalizer = 50
    validation_percentage = 0.02

    # Define how you load your training data
    use_generator = 1         # 1 if you want to use a generator, 0 if you want to load all your values at the beginning
    first_examples_bool = 0   # 1 to load the frames 0 to num_examples, 0 to load random num_examples in the num_timesteps frames
                              # Only useful if use_generator == 0
    # Define the validation dataset
    num_examples_val = 2         # Same as num_examples, must be < validation_percentage * num_timesteps
    first_examples_val_bool = 1  # Same as first_examples_bool

    # Define training parameters
    batch_size = 10            # Not too high 
    num_epochs = 10
    num_examples_per_epoch = 40000      # If use_generator == 0, number of examples that are loaded, else this helps define steps_per_epoch in the training
    num_examples_per_epoch = (num_examples_per_epoch//batch_size)*batch_size
    # Cannot be > 200 if use_generator == 0, RAM will crash

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
        data = tf.data.Dataset.from_generator(data_loader.get_data_with_generator, args=[num_examples_per_epoch, batch_size], output_types=((tf.float32, tf.float32, tf.int32, tf.float32), tf.float32), output_shapes = (((num_atoms, num_atom_features), (num_atoms, num_atoms, num_pairs_features), (num_atoms, max_num_of_bonds, 2), (num_atoms, max_num_of_bonds, 1)), (num_atoms, num_atoms)))
        print(data)
    else:
        X_input_atom, X_input_pairs, X_atom_graph, X_mask, Y = data_loader.get_data_no_generator(num_examples_per_epoch, first_examples_bool, 'train')
        print("X_input_atom shape:" + str(X_input_atom.shape))
        print("X_input_pairs shape: " + str(X_input_pairs.shape))
        print("X_atom_graph shape: " + str(X_atom_graph.shape))
        print("X_mask shape: " + str(X_mask.shape))
        print("Y shape: " + str(Y.shape))
    X_input_atom_val, X_input_pairs_val, X_atom_graph_val, X_mask_val, Y_val = data_loader.get_data_no_generator(num_examples_val, first_examples_val_bool, 'val')
    print("")
    print("X_input_atom_val shape:" + str(X_input_atom_val.shape))
    print("X_input_pairs_val shape: " + str(X_input_pairs_val.shape))
    print("X_atom_graph_val shape: " + str(X_atom_graph_val.shape))
    print("X_mask_val shape: " + str(X_mask_val.shape))
    print("Y_val shape: " + str(Y_val.shape))

    # Initialize model with inputs, and additionnal methods for training
    model = model_init(num_atoms, num_atom_features, num_pairs_features, max_num_of_bonds)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, mode='max', restore_best_weights=True)
    callbacks_list = [early_stopping]

    # Build optimizer by using adam optimizer and mse loss, accuracy metric 
    model.compile(optimizer='adam',
            loss='mse', metrics=['accuracy'])

    # Fit the model with the appropriated  data pipeline method,so the full dataset does not need to fit into memory.
    if use_generator == 1:
        history = model.fit(data.repeat().batch(batch_size=batch_size).prefetch(tf.data.experimental.AUTOTUNE), epochs=num_epochs, steps_per_epoch=int(num_examples_per_epoch/batch_size), 
            validation_data=([X_input_atom_val, X_input_pairs_val, X_atom_graph_val, X_mask_val], Y_val), callbacks=callbacks_list) 
    else:
        history = model.fit([X_input_atom, X_input_pairs, X_atom_graph, X_mask], Y, 
            batch_size=batch_size, epochs=num_epochs, 
            validation_data=([X_input_atom_val, X_input_pairs_val, X_atom_graph_val, X_mask_val], Y_val), callbacks=callbacks_list)

    #Save the model as my_model
    model.save('saved_model/my_secondmodel') 

    #Save  loss history into png file
    plot_hist(history,"output2_loss")

    # Run the KMC with the previous ML model
    num_iterations = 20
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
    molecules_of_interest = np.array([[4, 10, 3, 10, 0], [1, 4, 0, 4, 0]])  # with CH4H10_10ps
    # molecules_of_interest = np.array([[1, 4, 0, 4, 0]])  # with Features_CH4_600ps
    plot_molecules_of_interest(molecules_per_frame, molecule_list, time_range, molecules_per_frame_MD, molecule_list_MD, time_range_MD*0.012, molecules_of_interest) 
