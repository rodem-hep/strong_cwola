import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def reshape_sort_save(data_dir, raw_file, out_file):
    print(f"Converting {raw_file} -> {out_file}")
    print(" - Loading data")
    csts = pd.read_hdf(data_dir + raw_file).to_numpy().astype(np.float32)
    has_labels = csts.shape[1] % 3 == 1
    if has_labels:
        labels = csts[:, -1]
        csts = csts[:, :-1]
    else:
        labels = np.zeros_like(csts[:, 0])

    print(f" - Reshaping {csts.shape} -> ({csts.shape[0]}, -1, 3)")
    csts = csts.reshape(csts.shape[0], -1, 3)

    # Sort the constituents
    print(" - Sorting by Pt")
    order = np.argsort(-csts[..., 0], axis=-1)
    csts = np.take_along_axis(csts, order[..., None], axis=1)

    # Resave in a faster format for chunking
    print(" - Saving")
    with h5py.File(data_dir + out_file, "w") as file:
        file.create_dataset("csts", data=csts)
        file.create_dataset("labels", data=labels)


def main():
    data_dir = "/srv/beegfs/scratch/groups/rodem/LHCO/strong_cwola/"
    file_name = "events_anomalydetection_v2.h5"
    out_name = "pythia.h5"
    reshape_sort_save(data_dir, file_name, out_name)


if __name__ == "__main__":
    main()
