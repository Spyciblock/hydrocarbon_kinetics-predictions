import numpy as np
import networkx as nx
import pdb


class MoleculeList:
    def __init__(self, num_of_atoms, atom_types, num_of_types):
        self.num_of_atoms = num_of_atoms
        self.molecule_dict = np.array([])
        self.num_of_types = num_of_types
        if self.molecule_dict.shape[0] == 0:
            self.molecule_dict.shape = [0, self.num_of_types + self.num_of_types * (self.num_of_types + 1) // 2]
        self.atom_types = atom_types
        self.num_of_molecules_descriptors = self.num_of_types + self.num_of_types * (self.num_of_types + 1) // 2
        self.molecule_graph = nx.Graph(np.zeros([2, 2]))
        self.molecule_all_frames = np.zeros([0, 50000], dtype=int)
        self.step_with_reax = 0
        self.molecule_per_frame = np.zeros(self.molecule_dict.shape[0])

    def get_molecule_full(self, first_frame, bond_change):
        # Perform the whole analysis to get the numbers of molecules at all frames.
        self.create_graph(first_frame)
        [molecules_frame, molecules_counts] = self.get_molecules_frame(self.molecule_graph)
        self.update_molecule_per_frame_and_dict(molecules_frame, molecules_counts)
        self.molecule_all_frames = np.zeros([len(bond_change)+1, 50000], dtype=int)
        self.molecule_all_frames[0, :self.molecule_per_frame.shape[0]] = self.molecule_per_frame
        step_with_reax = 1
        time_counter = 100
        for frame in bond_change.keys():
            if frame > time_counter:
                print(frame, flush=True)
                time_counter += 100
            [molecules_frame_change, molecules_counts_change, old_molecules_frame, old_molecule_bond_change, new_molecules_frame, new_molecule_bond_change] = self.get_molecules_change(bond_change[frame])
            self.update_molecule_per_frame_and_dict(molecules_frame_change, molecules_counts_change)
            self.update_molecule_all_frames(step_with_reax)
            step_with_reax += 1
        self.molecule_all_frames = self.molecule_all_frames[:, 0:self.molecule_dict.shape[0]]
        return self.molecule_all_frames, self.molecule_dict

    def initialize_molecules(self, first_frame, start_frame_MD):
        self.create_graph(first_frame)
        if self.reaction_features[1] == 1:
            self.update_molecules_idx_atoms_frame()
        self.time_range = np.array([start_frame_MD])
        [molecules_frame, molecules_counts] = self.get_molecules_frame(self.molecule_graph)
        self.update_molecule_per_frame_and_dict(molecules_frame, molecules_counts)
        self.molecule_all_frames[0, :self.molecule_per_frame.shape[0]] = self.molecule_per_frame
        self.step_with_reax = 1

    def update_after_reaction(self, frame, bond_change_frame):
        self.time_range = np.append(self.time_range, frame)
        [molecules_frame_change, molecules_counts_change, old_molecules_frame, old_molecule_bond_change, new_molecules_frame, new_molecule_bond_change] = self.get_molecules_change(bond_change_frame)
        self.update_molecule_per_frame_and_dict(molecules_frame_change, molecules_counts_change)
        self.update_molecule_all_frames(self.step_with_reax)
        self.step_with_reax += 1
        return [old_molecules_frame, old_molecule_bond_change, new_molecules_frame, new_molecule_bond_change]

    def create_graph(self, bond_matrix):
        # Create the graph of molecules
        self.molecule_graph = nx.Graph(bond_matrix)
        for atom in range(self.num_of_atoms):
            self.molecule_graph.nodes[atom]['type'] = self.atom_types[atom]
        for edge in self.molecule_graph.edges:
            self.molecule_graph.edges[edge]['type'] = self.get_edge_type(edge)

    def get_molecules_frame(self, molecule_graph, atoms_involved=np.array([])):
        # Get the molecules at one specific frame
        molecules_frame_graph = nx.connected_components(molecule_graph)
        molecules = np.zeros([nx.number_connected_components(molecule_graph), self.num_of_molecules_descriptors])
        mol_num = 0
        if atoms_involved.shape[0] != 0:
            mol_atoms_involved = np.zeros(atoms_involved.shape, dtype=int)
        for mol in molecules_frame_graph:
            mol_graph = self.molecule_graph.subgraph(mol)
            [types_n, counts_n] = np.unique(list(nx.get_node_attributes(mol_graph, 'type').values()), return_counts=True)
            [types_e, counts_e] = np.unique(list(nx.get_edge_attributes(mol_graph, 'type').values()), return_counts=True)
            if atoms_involved.shape[0] != 0:
                mol_atoms_involved[np.where(np.in1d(atoms_involved, list(mol_graph)))] = mol_num
            molecules[mol_num, types_n - 1] = counts_n
            if types_e.shape[0] > 0:
                molecules[mol_num, types_e + self.num_of_types] = counts_e
            mol_num += 1
        [molecules_frame, molecules_frame_count] = np.unique(molecules, return_counts=True, axis=0)
        if atoms_involved.shape[0] != 0:
            return[molecules_frame, molecules_frame_count, molecules, mol_atoms_involved]
        return [molecules_frame, molecules_frame_count]

    def get_edge_type(self, edge):
        # Get the type of an edge, depending on the type of the atoms it is bonding.
        m1 = min(self.molecule_graph.nodes[edge[0]]['type'], self.molecule_graph.nodes[edge[1]]['type'])
        m2 = max(self.molecule_graph.nodes[edge[0]]['type'], self.molecule_graph.nodes[edge[1]]['type'])
        return (m1 - 1) * self.num_of_types - m1 * (m1 - 1) // 2 + m2 - 1

    def update_molecule_per_frame_and_dict(self, molecules_frame, molecules_count):
        # Given the molecules at a frame and their counts, update molecule_per_frame and molecule_dict.
        for mol in range(molecules_frame.shape[0]):
            molecule_idx = np.where((self.molecule_dict == molecules_frame[mol, :]).all(axis=1))
            if molecule_idx[0].shape[0] == 0:
                self.molecule_dict = np.vstack([self.molecule_dict, molecules_frame[mol,:]])
                self.molecule_per_frame = np.append(self.molecule_per_frame, molecules_count[mol])
            elif molecule_idx[0].shape[0] == 1:
                self.molecule_per_frame[molecule_idx[0]] += molecules_count[mol]

    def get_molecules_change(self, bond_change):
        # Compute the change in molecules.
        atoms_involved = np.unique(bond_change[:, :2])
        molecule_subset_old = set()
        molecule_subset_new = set()
        for atom in atoms_involved:
            molecule_subset_old = molecule_subset_old.union(nx.node_connected_component(self.molecule_graph, atom))
        [old_molecules, old_molecules_count, old_molecules_frame, old_mol_atoms_involved] = self.get_molecules_frame(self.molecule_graph.subgraph(molecule_subset_old), atoms_involved)
        old_molecule_bond_change = self.get_molecule_bond_change(old_mol_atoms_involved, atoms_involved, bond_change)
        self.update_molecule_graph(bond_change)
        for atom in atoms_involved:
            molecule_subset_new = molecule_subset_new.union(nx.node_connected_component(self.molecule_graph, atom))
        [new_molecules, new_molecules_count, new_molecules_frame, new_mol_atoms_involved] = self.get_molecules_frame(self.molecule_graph.subgraph(molecule_subset_new), atoms_involved)
        new_molecule_bond_change = self.get_molecule_bond_change(new_mol_atoms_involved, atoms_involved, bond_change)
        molecules_frame_change = new_molecules.copy()
        molecules_counts_change = new_molecules_count.copy()
        for mol in range(old_molecules.shape[0]):
            molecule_idx = np.where((molecules_frame_change == old_molecules[mol, :]).all(axis=1))
            if molecule_idx[0].shape[0] == 0:
                molecules_frame_change = np.vstack([molecules_frame_change, old_molecules[mol, :]])
                molecules_counts_change = np.append(molecules_counts_change, -old_molecules_count[mol])
            elif molecule_idx[0].shape[0] == 1:
                molecules_counts_change[molecule_idx[0]] -= old_molecules_count[mol]
        return [molecules_frame_change, molecules_counts_change, old_molecules_frame, old_molecule_bond_change, new_molecules_frame, new_molecule_bond_change]

    def update_molecule_graph(self, bond_change):
        # Update the graph of molecules.
        for i in range(bond_change.shape[0]):
            if bond_change[i, 2] == 1:
                self.molecule_graph.add_edge(bond_change[i, 0], bond_change[i, 1], type=self.get_edge_type((bond_change[i, 0], bond_change[i, 1])))
            if bond_change[i, 2] == 0:
                self.molecule_graph.remove_edge(bond_change[i, 0], bond_change[i, 1])

    def update_molecules_idx_atoms_frame(self):
        molecules_frame_graph = nx.connected_components(self.molecule_graph)
        mol_num = 0
        for mol in molecules_frame_graph:
            mol_graph = self.molecule_graph.subgraph(mol)
            self.molecules_idx_atoms[list(mol_graph), self.step_with_reax] = mol_num
            mol_num += 1

    def update_molecule_all_frames(self, step_with_reax):
        # Update the list of number of all molecules at all frames.
        if self.molecule_all_frames.shape[1] < self.molecule_per_frame.shape[0]:
            store_molecule_all_frames = self.molecule_all_frames.copy()
            self.molecule_all_frames = np.zeros([store_molecule_all_frames.shape[0], 2*store_molecule_all_frames.shape[1]], dtype=int)
            self.molecule_all_frames[0:store_molecule_all_frames.shape[0], 0:store_molecule_all_frames.shape[1]] = store_molecule_all_frames
        if step_with_reax >= self.molecule_all_frames.shape[0]:
            # pdb.set_trace()
            store_molecule_all_frames = self.molecule_all_frames.copy()
            self.molecule_all_frames = np.zeros(
                [2*store_molecule_all_frames.shape[0], store_molecule_all_frames.shape[1]], dtype=int)
            self.molecule_all_frames[0:store_molecule_all_frames.shape[0],
                0:store_molecule_all_frames.shape[1]] = store_molecule_all_frames
        self.molecule_all_frames[step_with_reax, 0:self.molecule_per_frame.shape[0]] = self.molecule_per_frame

    def get_molecule_bond_change(self, mol_atoms_involved, atoms_involved, bond_change):
        molecule_bond_change = bond_change.copy()
        for i in range(molecule_bond_change.shape[0]):
            molecule_bond_change[i, :2] = mol_atoms_involved[np.where(atoms_involved[:, None] == bond_change[i, :2])[0]]
        return molecule_bond_change

