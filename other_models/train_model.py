import boto3
import numpy as np
import pickle
import pdb
from scipy import sparse
import tensorflow as tf
import os, datetime

from data_generator import DataLoader
from data_generator_bucket import DataLoaderBucket
from kinetic_monte_carlo import run_KMC
from utils import *
from molecules import MoleculeList
from model import model_init

#Code to train our model
# Used to plot loss and accuracy evolution
def plot_hist(history, imgname):
    os.mkdir("./plots/"+imgname)
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
    plt.legend(loc='upper right')
    plt.savefig("./plots/" + str(imgname) + "/loss.png")
    plt.clf()

    plt.figure(2)
    plt.plot(prediction_probs_loss, label='Training prediction probs loss')
    plt.plot(val_prediction_probs_loss, label='Validation prediction probs loss')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Training and Validation prediction probs loss')
    plt.savefig("./plots/" + str(imgname) + "/probsloss.png")
    plt.clf()

    plt.figure(3)
    plt.plot(prediction_time_loss, label='Training prediction time loss')
    plt.plot(val_prediction_time_loss, label='Validation prediction time loss')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Training and Validation prediction time loss')
    plt.savefig("./plots/" + str(imgname) + "/timeloss.png")
    plt.clf()
    plt.close()
    history.history['loss'] = []
    history.history['val_loss'] = []

    history.history['prediction_probs_loss'] = []
    history.history['val_prediction_probs_loss'] = []

    history.history['prediction_time_loss'] = []
    history.history['val_prediction_time_loss'] = []


if __name__ == "__main__":
    use_bucket = 0

    # Select the way to access data: from bucket or disk storage
    if use_bucket == 0:
        foldername = '../Data/C4H10_600ps/'
    else:
        foldername = 'C4H10_10ps/'
    # Define which features to use with boolean value
    atom_features_bool = {"Atom types": 1,
                          "Cycle size": 1,
                          "Molecule size": 1}
    pairs_features_bool = {"Bonded": 1,
                           "Same cycle": 1,
                           "Same molecule": 1}
    max_num_of_bonds = 10

    # Get useful constants used for the model
    num_atoms, num_atom_features, num_pairs_features, num_timesteps = get_constants(foldername, atom_features_bool,
                                                                                    pairs_features_bool)

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
    use_generator = 1  # 1 if you want to use a generator, 0 if you want to load all your values at the beginning
    first_examples_bool = 1  # 1 to load the frames 0 to num_examples, 0 to load random num_examples in the num_timesteps frames
    # Only useful if use_generator == 0
    # Define the validation dataset
    num_examples_val = 10  # Same as num_examples, must be < validation_percentage * num_timesteps
    first_examples_val_bool = 1  # Same as first_examples_bool

    # Define training parameters
    batch_size = 10  # Not too high
    num_epochs = 5
    num_examples_per_epoch = 5000  # If use_generator == 0, number of examples that are loaded, else this helps define steps_per_epoch in the training
    num_examples_per_epoch = (num_examples_per_epoch // batch_size) * batch_size
    # Cannot be > 200 if use_generator == 0, RAM will crash
    depth = 5
    myLosses = {'prediction_probs': "categorical_crossentropy", 'prediction_time': 'mse'}#categorical_crossentropy,binary_crossentropy
    myLossesWeights = {'prediction_probs': 10 ** 5, 'prediction_time': 10 ** -13}
    hidden_size = 32
    lr = 0.001
    nbdense = 1

    # Select how to load and how to preprocess data according to the path selected,using data generator or not that fits with our computational ressources
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
        data = tf.data.Dataset.from_generator(data_loader.get_data_with_generator,
                                              args=[num_examples_per_epoch, batch_size], output_types=(
                (tf.float32, tf.float32, tf.int32, tf.float32, tf.int32), (tf.float32, tf.float32)), output_shapes=(((
                                                                                                                         num_atoms,
                                                                                                                         num_atom_features),
                                                                                                                     (
                                                                                                                         num_atoms * (
                                                                                                                                 num_atoms - 1) // 2,
                                                                                                                         num_pairs_features),
                                                                                                                     (
                                                                                                                         num_atoms,
                                                                                                                         max_num_of_bonds,
                                                                                                                         2),
                                                                                                                     (
                                                                                                                         num_atoms,
                                                                                                                         max_num_of_bonds,
                                                                                                                         1),
                                                                                                                     (
                                                                                                                         num_atoms * (
                                                                                                                                 num_atoms - 1) // 2,
                                                                                                                         3)),
                                                                                                                    ((
                                                                                                                             num_atoms * (
                                                                                                                             num_atoms - 1) // 2),
                                                                                                                     1)))
    else:
        X_input_atom, X_input_pairs, X_atom_graph, X_mask, X_extract_pairs, Y, Y_time = data_loader.get_data_no_generator(
            num_examples_per_epoch, first_examples_bool, 'train')
        print("X_input_atom shape:" + str(X_input_atom.shape))
        print("X_input_pairs shape: " + str(X_input_pairs.shape))
        print("X_atom_graph shape: " + str(X_atom_graph.shape))
        print("X_mask shape: " + str(X_mask.shape))
        print("X_extract_pairs shape:" + str(X_extract_pairs.shape))
        print("Y shape: " + str(Y.shape))
        print("Y_time shape: " + str(Y_time.shape))
    X_input_atom_val, X_input_pairs_val, X_atom_graph_val, X_mask_val, X_extract_pairs_val, Y_val, Y_time_val = data_loader.get_data_no_generator(
        num_examples_val, first_examples_val_bool, 'val')
    print("")
    print("X_input_atom_val shape:" + str(X_input_atom_val.shape))
    print("X_input_pairs_val shape: " + str(X_input_pairs_val.shape))
    print("X_atom_graph_val shape: " + str(X_atom_graph_val.shape))
    print("X_mask_val shape: " + str(X_mask_val.shape))
    print("X_extract_pairs_val shape:" + str(X_extract_pairs_val.shape))
    print("Y_val shape: " + str(Y_val.shape))
    print("Y_time_val shape: " + str(Y_time_val.shape))

    # Initialize model with inputs, and additionnal methods for training
    tf.keras.backend.clear_session()
    model = model_init(num_atoms, num_atom_features, num_pairs_features, max_num_of_bonds, depth, lr, hidden_size,
                       nbdense)
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, mode='max', restore_best_weights=False)
    # callbacks_list = [early_stopping]

    # Build the model by using the adam optimizer and mse loss, accuracy metric
    optim = tf.keras.optimizers.Adam(learning_rate=ratelist[lr])
    model.compile(optimizer=optim,
                  loss=myLosses, loss_weights=myLossesWeights, metrics=['accuracy'])

    # Fit the model with the appropriated  data pipeline method,so the full dataset does not need to fit into memory.
    if use_generator == 1:
        history = model.fit(data.repeat().batch(batch_size=batch_size).prefetch(tf.data.experimental.AUTOTUNE),
                            epochs=num_epochs, steps_per_epoch=int(num_examples_per_epoch / batch_size),
                            validation_data=(
                                [X_input_atom_val, X_input_pairs_val, X_atom_graph_val, X_mask_val,
                                 X_extract_pairs_val],
                                [Y_val, Y_time_val]), callbacks=None)
    else:
        history = model.fit([X_input_atom, X_input_pairs, X_atom_graph, X_mask, X_extract_pairs], [Y, Y_time],
                            batch_size=batch_size, epochs=num_epochs,
                            validation_data=(
                                [X_input_atom_val, X_input_pairs_val, X_atom_graph_val, X_mask_val,
                                 X_extract_pairs_val],
                                [Y_val, Y_time_val]), callbacks=None)

    # Save the model
    savedir = '16112020model_' + 'depth' + str(depth) + "_hsize" + str(hidden_size) + "_lr-" + str(lr)
    if not os.path.exists("./plots/" + str(savedir)):
        os.mkdir("./plots/" + str(savedir))
    model.save('./plots/' + str(savedir))  # vary learning rate
    # Save  loss history into png file
    plot_hist(history, savedir)
        # Run the KMC with the ML model
    num_iterations = 50
    bond_change, first_frame, time_range = run_KMC(model, num_iterations, foldername, atom_features_bool,
                                                       pairs_features_bool,
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
    plot_molecules_of_interest(molecules_per_frame, molecule_list, time_range, molecules_per_frame_MD, molecule_list_MD,
                               time_range_MD * 0.012, molecules_of_interest, savedir)
