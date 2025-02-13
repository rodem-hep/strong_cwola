import h5py
import numpy as np


def split_sig_bkg(data_dir, file_name):
    with h5py.File(data_dir + file_name, "r") as f:
        data = {key: f[key][:].astype(np.float32) for key in f}

    # Create the mask based on the labels
    mask = data["labels"].astype(bool)

    # Split the data based on the mask and resave
    with h5py.File(data_dir + "sig_" + file_name, "w") as f:
        for key in data:
            f.create_dataset(key, data=data[key][mask])

    with h5py.File(data_dir + "bkg_" + file_name, "w") as f:
        for key in data:
            f.create_dataset(key, data=data[key][~mask])


def main():
    data_dir = "/srv/beegfs/scratch/groups/rodem/LHCO/strong_cwola/"
    file_name = "clustered_pythia.h5"
    split_sig_bkg(data_dir, file_name)


if __name__ == "__main__":
    main()
