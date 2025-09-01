import numpy as np
import re
import csv, sys

################################################################################
#                           PARSING FUNCTIONS
################################################################################

def parse_lammps_velocities(filename, stride=1):
    """
    Generator that yields (timestep, data) for each block in the LAMMPS dump file.
    'data' is a list of tuples: (atom_id, atom_type, vx, vy, vz).

    Only yields every `stride`-th frame from the file. For example,
    if stride=4, it yields frames 0, 4, 8, ...
    """
    frame_index = 0

    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break  # end of file

            if "ITEM: TIMESTEP" in line:
                # read next line => the timestep
                timestep_line = f.readline()
                if not timestep_line:
                    break
                timestep = int(timestep_line.strip())

                # next line => ITEM: NUMBER OF ATOMS
                line = f.readline()
                num_atoms_line = f.readline()  # read the actual number
                if not num_atoms_line:
                    break
                num_atoms = int(num_atoms_line.strip())

                # skip BOX BOUNDS lines
                boxbounds_line = f.readline()  # e.g. 'ITEM: BOX BOUNDS...'
                xbounds = f.readline()
                ybounds = f.readline()
                zbounds = f.readline()

                # next line => ITEM: ATOMS id type x y z vx vy vz
                line = f.readline()
                # now read `num_atoms` lines
                atoms_data = []
                for _ in range(num_atoms):
                    atom_line = f.readline().split()
                    atom_id = int(atom_line[0])
                    atom_type = int(atom_line[1])
                    vx = float(atom_line[5])
                    vy = float(atom_line[6])
                    vz = float(atom_line[7])
                    atoms_data.append((atom_id, atom_type, vx, vy, vz))

                # Decide if we yield this frame or skip it
                if frame_index % stride == 0:
                    yield (timestep, atoms_data)

                frame_index += 1


################################################################################
#                          DATA-STRUCTURE BUILDERS
################################################################################

def build_all_vels_dict(parser_func, filename):
    """
    Iterate through all timesteps in a LAMMPS dump and build a dictionary:

        all_vels = {
           atom_id: np.array of shape (n_frames, 3)
        }

    where each array is the velocity trajectory for that atom (in A/fs, if
    dividing the default A/ps by 1000).
    """
    all_vels = {}   # key: atom_id, value: list of velocity 3-vectors over time
    timesteps = []

    for (timestep, atoms_data) in parser_func(filename):
        timesteps.append(timestep)
        for (atom_id, atom_type, vx, vy, vz) in atoms_data:
            # Convert velocities from A/ps to A/fs (divide by 1000 if needed)
            vx_fs = vx / 1000.0
            vy_fs = vy / 1000.0
            vz_fs = vz / 1000.0

            if atom_id not in all_vels:
                all_vels[atom_id] = []
            all_vels[atom_id].append([vx_fs, vy_fs, vz_fs])

    # Convert lists to NumPy arrays
    for atom_id in all_vels:
        all_vels[atom_id] = np.array(all_vels[atom_id], dtype=np.float64)

    timesteps_sorted = sorted(timesteps)
    return all_vels, timesteps_sorted

################################################################################
#                          VACF AND POWER SPECTRUM
################################################################################

def compute_vacf(all_vels, max_lag=None):
    """
    Compute the ensemble-averaged velocity autocorrelation function (VACF).
   
    Parameters:
        all_vels (dict): {atom_id: (n_frames, 3) np.array of velocities}
        max_lag (int, optional): maximum lag time for VACF. 
            Defaults to n_frames//2 if None.
       
    Returns:
        vacf (np.ndarray): Normalized VACF of shape (max_lag,).
    """
    n_frames = next(iter(all_vels.values())).shape[0]
    if max_lag is None:
        max_lag = n_frames // 2

    vacf = np.zeros(max_lag, dtype=np.float64)

    # Convert dictionary to an array: shape (n_atoms, n_frames, 3)
    atom_vels = np.stack([all_vels[aid] for aid in all_vels])  # (n_atoms, n_frames, 3)
    n_atoms = atom_vels.shape[0]

    for tau in range(max_lag):
        # dot-products: v(t) · v(t+tau)
        dot_products = np.sum(
            atom_vels[:, :n_frames - tau, :] * atom_vels[:, tau:n_frames, :],
            axis=-1
        )  # shape: (n_atoms, n_frames - tau)
        vacf[tau] = np.sum(dot_products) / (n_atoms * (n_frames - tau))

    # Normalize so that vacf[0] = 1.0 (assuming vacf[0] != 0)
    vacf /= vacf[0]
    return vacf



################################################################################
#                           EXAMPLE MAIN USAGE
################################################################################

if __name__ == "__main__":
    """
    Usage:
        python script.py <lammps_dump_file> <spectrum_output.csv> <vacf_output.csv>

    - <lammps_dump_file>: a LAMMPS velocity dump
    - <spectrum_output.csv>: file to store freq, real part, imag part, power spectrum
    - <vacf_output.csv>: file to store the computed VACF vs time
    """
    import csv

    # Command-line args
    lammps_filename = sys.argv[1]
    vacf_csv = sys.argv[2]

    #----------------------------------------------------------------
    # 1) Parse classical LAMMPS velocities and build dict
    #----------------------------------------------------------------
    all_vels, timesteps = build_all_vels_dict(parse_lammps_velocities, lammps_filename)

    # Suppose each frame is 1.25 fs apart (example)
    dt_fs = 0.2
    temperature = 300.0  # K

    #----------------------------------------------------------------
    # 2) Compute the VACF
    #----------------------------------------------------------------
    vacf = compute_vacf(all_vels, max_lag=None)

    #----------------------------------------------------------------
    # 2a) SAVE the VACF to a file
    #----------------------------------------------------------------
    with open(vacf_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_fs", "VACF"])
        for i, val in enumerate(vacf):
            t_fs = i * dt_fs
            writer.writerow([t_fs, val])
    print(f"VACF saved to {vacf_csv}")


