import argparse
import logging
from pathlib import Path

import awkward as ak
import fastjet
import h5py
import numpy as np
import rootutils
import vector

root = rootutils.setup_root(search_from=__file__, pythonpath=True)


from src.data.clustering import (
    cluster_numpy_batch,
    convert_to_relative,
    make_padded,
    make_ragged,
    n_subjettiness,
)

vector.register_awkward()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


def none_or_int(x: str) -> int | None:
    if x.lower() == "none":
        return None
    return int(x)


def str_to_bool(x: str) -> bool:
    return x.lower() in {"t", "true", "y", "yes", "1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster analysis parameters",
    )
    parser.add_argument(
        "--R",
        type=float,
        default=1.0,
        help="Anti-kt Radius parameter",
    )
    parser.add_argument(
        "--pt_min",
        type=float,
        default=3,
        help="Minimum pt",
    )
    parser.add_argument(
        "--n_events",
        type=none_or_int,
        help="Number of events (None for all)",
        default=1000,
    )
    parser.add_argument(
        "--n_csts",
        type=int,
        default=200,
        help="Maximum number of constituents per jet",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="pythia.h5",
        help="Path to the data file",
    )
    parser.add_argument(
        "--do_subclustering",
        type=str_to_bool,
        default=True,
        help="If set, perform subclustering and calculate the tau variables",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default="/srv/beegfs/scratch/groups/rodem/LHCO/strong_cwola/",
        help="Path to the output file",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load all data for the clustering - trim to the number of events and csts
    in_path = args.data_dir / args.input_file
    log.info(f"Loading data from {in_path}")
    with h5py.File(in_path, "r") as file:
        csts = file["csts"][: args.n_events]  # Must cluster with all constituents
        labels = file["labels"][: args.n_events]

    log.info(f"Clustering each event using anti-kt with R={args.R}")
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, args.R)
    clusters = cluster_numpy_batch(csts, jetdef, mask=csts[..., 0] > 0)

    # We are only interested in the leading two jets per event
    log.info("Extracting the leading two jets")
    jets = clusters.inclusive_jets()[:, -2:]  # Output is increasing in PT
    csts = clusters.constituents()[:, -2:]

    # The actual data from the jets we want to save
    jet_data = [jets.pt, jets.eta, jets.phi, jets.m]

    # Perform subclustering if requested
    if args.do_subclustering:
        # Subclustering will fail if there are less consituents than numjets
        # To prevent this we pad each jet with up to 3 zero-energy constituents
        # The corresponding clusters will be dropped later
        zero_cst = ak.zip(
            {"pt": 0, "eta": 0, "phi": 0, "mass": 0}, with_name="Momentum4D"
        )
        x = ak.pad_none(csts, 3, axis=2)
        x = ak.fill_none(x, zero_cst)

        log.info("Subclustering the leading jets with GenKT")
        subjetdef = fastjet.JetDefinition(fastjet.genkt_algorithm, 1.0, 1.0)
        subjets = fastjet.ClusterSequence(x, subjetdef)

        log.info("Subclustering using njet = 1...")
        subjets1 = subjets.exclusive_jets(n_jets=1)
        log.info("Subclustering using njet = 2...")
        subjets2 = subjets.exclusive_jets(n_jets=2)
        log.info("Subclustering using njet = 3...")
        subjets3 = subjets.exclusive_jets(n_jets=3)

        # Drop the jets that have 0 energy - created by the padding
        subjets2 = make_ragged(subjets2, subjets2.pt > 0, axis=2)
        subjets3 = make_ragged(subjets3, subjets3.pt > 0, axis=2)

        # Calculate the tau variables
        log.info("Calculating tau1...")
        tau1 = n_subjettiness(subjets1, csts, args.R)
        log.info("Calculating tau2...")
        tau2 = n_subjettiness(subjets2, csts, args.R)
        log.info("Calculating tau3...")
        tau3 = n_subjettiness(subjets3, csts, args.R)
        jet_data += [tau1, tau2, tau3]  # Make sure it is saved

    log.info("Combining the jet observables")
    jet_obs = ak.concatenate([j[..., None] for j in jet_data], axis=-1)
    jet_obs = jet_obs.to_numpy()
    jet1_obs = jet_obs[:, -1]  # Output is increasing in PT -> last jet is leading
    jet2_obs = jet_obs[:, -2]

    log.info("Converting the constituent observables to relative coordinates")
    jet1_cnsts = make_padded(csts[:, -1], args.n_csts)[..., :3]  # Massless
    jet2_cnsts = make_padded(csts[:, -2], args.n_csts)[..., :3]
    convert_to_relative(jet1_cnsts, jet1_obs)
    convert_to_relative(jet2_cnsts, jet2_obs)

    log.info("Calculating the invariant mass of the leading two jets")
    mjj = ((jets[:, -1] + jets[:, -2]).m).to_numpy()[:, None]  # N x 1

    log.info("Creating an event idx variable")
    event_ids = np.arange(jet1_obs.shape[0], dtype=np.int32)[:, None]  # N x 1

    out_path = args.data_dir / ("clustered_" + args.input_file)
    log.info(f"Saving all data to {out_path}")
    with h5py.File(out_path, "w") as file:
        dtype = np.dtype("float32")
        file.create_dataset("jets1", data=jet1_obs, dtype=dtype)
        file.create_dataset("jets2", data=jet2_obs, dtype=dtype)
        file.create_dataset("csts1", data=jet1_cnsts, dtype=dtype)
        file.create_dataset("csts2", data=jet2_cnsts, dtype=dtype)
        file.create_dataset("mjj", data=mjj, dtype=dtype)
        file.create_dataset("labels", data=labels, dtype=bool)
        file.create_dataset("event_ids", data=event_ids, dtype=np.int32)


if __name__ == "__main__":
    main()
