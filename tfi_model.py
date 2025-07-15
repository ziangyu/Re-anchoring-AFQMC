import numpy as np
import copy

from tfi_utils import sigma_x, sigma_z

class TransverseFieldIsing:
    def __init__(self, magnetic_field, num_spins_x, num_spins_y, boundary_x = 'p', boundary_y = 'p') -> None:
        """
        The Transverse-Field Ising model. 
        
        magnetic_field: strength of the transverse field
        num_spins_x: number of spins in x-axis. 
        num_spins_y: number of spins in y-axis. 
            for 1D model, set one of them to be 1. 
        boundary_x: boundary condition for x-axis, either 'p' (periodic, default) or 'o' (open)
        boundary_y: boundary condition for y-axis, either 'p' (periodic, default) or 'o' (open)
        """
        self.magnetic_field = magnetic_field  
        self.num_spins_x = num_spins_x             
        self.num_spins_y = num_spins_y
        self.boundary_x = boundary_x
        self.boundary_y = boundary_y

        self.num_spins = num_spins_x * num_spins_y   # total number of spins
        

    def get_bond_set(self):
        """
        Store the physical bonds of the model

        Returns
        bond_set: numpy array of shape [num_bonds, 2]
            each entry [i, j] represents a bond between spin i and j (1-based indexing)
        """
        bond_set = []

        # Build snake-like lattice indexing (zigzag rows)
        lattice = np.zeros([self.num_spins_x, self.num_spins_y], dtype=int)
        count = 1
        for i in range(self.num_spins_x):
            row = np.arange(count, count + self.num_spins_y)
            if i % 2 == 1:
                row = row[::-1]
            lattice[i, :] = row
            count += self.num_spins_y

        for i in range(self.num_spins_x):
            if self.num_spins_y % 2 == 0:
                for j in range(int(self.num_spins_y / 2)):
                    bond_set.append([lattice[i, j * 2], lattice[i, j * 2 + 1]])
                for j in range(int(self.num_spins_y / 2) - 1):
                    bond_set.append([lattice[i, j * 2 + 1], lattice[i, j * 2 + 2]])
                if self.boundary_y == 'p' and self.num_spins_y > 1:
                    bond_set.append([lattice[i, 0], lattice[i, -1]])

            else:
                for j in range(int(self.num_spins_y / 2)):
                    bond_set.append([lattice[i, j * 2], lattice[i, j * 2 + 1]])
                if self.boundary_y == 'p' and self.num_spins_y > 1:
                    bond_set.append([lattice[i, 0], lattice[i, -1]])
                for j in range(int(self.num_spins_y / 2)):
                    bond_set.append([lattice[i, j * 2 + 1], lattice[i, j * 2 + 2]])

        
        for j in range(self.num_spins_y):
            if self.num_spins_x % 2 == 0:
                for i in range(int(self.num_spins_x / 2)):
                    bond_set.append([lattice[i * 2, j], lattice[i * 2 + 1, j]])
                for i in range(int(self.num_spins_x / 2) - 1):
                    bond_set.append([lattice[i * 2 + 1, j], lattice[i * 2 + 2, j]])
                if self.boundary_x == 'p' and self.num_spins_x > 1:
                        bond_set.append([lattice[0, j], lattice[-1, j]])

            else:
                for i in range(int(self.num_spins_x / 2)):
                    bond_set.append([lattice[i * 2, j], lattice[i * 2 + 1, j]])
                if self.boundary_x == 'p' and self.num_spins_x > 1:
                    bond_set.append([lattice[0, j], lattice[-1, j]])
                for i in range(int(self.num_spins_x / 2)):
                    bond_set.append([lattice[i * 2 + 1, j], lattice[i * 2 + 2, j]])

        bond_set = np.array(bond_set)

        return bond_set

    def get_h_phi(self, phi):
        """
        Apply the Hamiltonian H to a wavefunction phi represented in TT format

        Parameters
        phi: list of numpy arrays
            The TT representation of a many-body wavefunction
            Each core should be of shape (r_left, 2, r_right), except possibly the first and last cores

        Returns
        h_phi: list of numpy arrays
            A list of TT cores representing the action H phi.
            Each core has shape (num_terms, r_left, 2, r_right),
            where num_terms = num_spin + num_bond (i.e. number of one-body + two-body terms).
        """
        # make a deep copy to avoid modifying the original phi
        phi = copy.deepcopy(phi)

        # store the physical bonds
        bond_set = self.get_bond_set()
        num_bonds = len(bond_set)

        # ensure boundary TT cores have shape (1, 2, r) or (r, 2, 1)
        if len(phi[0].shape) == 2:
            phi[0] = phi[0][np.newaxis, :, :]
        if len(phi[-1].shape) == 2:
            phi[-1] = phi[-1][:, :, np.newaxis]

        # initialize h_phi[i] as a stack of identical copies of phi[i]
        # shape: [num_terms, r_left, 2, r_right]
        num_terms = self.num_spins + num_bonds
        h_phi = [np.tile(core, (num_terms, 1, 1, 1)) for core in phi]

        # apply one-body term: -h sum_i sigma^x_i
        # only the i-th core is modified at the i-th "term index"
        for d in range(self.num_spins):
            h_phi[d][d] = np.einsum('ij, pir -> pjr', -self.magnetic_field * sigma_x, phi[d])

        # apply two-body interaction term: -sigma^z_i sigma^z_j
        # each bond contributes to a new "term" starting at index self.num_spin
        for b in range(num_bonds):
            bond = bond_set[b]
            i = bond[0] - 1  # convert 1-based to 0-based index
            j = bond[1] - 1
            term_index = self.num_spins + b  # index in the [num_terms,...] stack

            # act -sigma^z on site i and +sigma^z on site j
            h_phi[i][term_index] = np.einsum('ij, pir -> pjr', -sigma_z, phi[i])
            h_phi[j][term_index] = np.einsum('ij, pir -> pjr', sigma_z, phi[j])

        return h_phi
