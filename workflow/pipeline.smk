from pathlib import Path

########################################


# Load a yaml file as the config
configfile: "configs/pipeline.yaml"


# Required for running on apptainer
container: config["container"]


# Check if these variables exist before attempting to run
envvars:
    "WANDB_API_KEY",


# Wildcards not allowed underscores
wildcard_constraints:
    gen="[0-9a-zA-Z]+",
    fold="[0-9]+",
    seed="[0-9]+",


# Important paths become Paths
data_dir = Path(config["data_dir"])
folds = range(config["num_folds"])
seeds = range(config["num_seeds"])

########################################


# rule all:
#     input:
#         pythia_files=expand(
#             data_dir
#             / config["project_name"]
#             / "pythia_{dope}_fold_{fold}_seed_{seed}",
#             dope=config["dope"],
#             fold=folds,
#             seed=seeds,
#         ),
#         herwig_files=expand(
#             data_dir / config["project_name"] / "herwig_fold_{fold}_seed_{seed}",
#             fold=folds,
#             seed=seeds,
#         ),


# rule train:
#     output:
#         data_dir / config["project_name"] / "{gen_dope}_fold_{fold}_seed_{seed}" /
#     input:
#         sig_file=data_dir / "clustered_pythia_sig.h5",
#         bkg_file=lambda w: data_dir
#         / (
#             "clustered_herwig.h5"
#             if w.gen_dope == "herwig"
#             else "clustered_pythia_bkg.h5"
#         ),
#     params:
#         output_dir=data_dir,
#         project_name=config["project_name"],
#         n_sig=90_000,
#         n_bkg=1000_000,
#         n_dope=lambda w: 0 if w.gen_dope == "herwig" else int(w.gen_dope.split("_")[1]),
#         seed=lambda w: w.seed,
#         test_fold=lambda w: w.fold,
#         num_folds=config["num_folds"],
#         model_name=lambda w: f"{w.gen_dope}_fold_{w.fold}_seed_{w.seed}",
#         extra_test=lambda w: (
#             f"clustered_pythia_bkg.h5"
#             if w.gen_dope == "herwig"
#             else "clustered_herwig.h5"
#         ),
#     shell:
#         """
#         python scripts/train.py \
#         model_name={params.model_name} \
#         output_dir={params.output_dir} \
#         project_name={params.project_name} \
#         seed={params.seed} \
#         datamodule.n_sig={params.n_sig} \
#         datamodule.n_bkg={params.n_bkg} \
#         datamodule.n_dope={params.n_dope} \
#         datamodule.test_fold={params.test_fold} \
#         datamodule.num_folds={params.num_folds} \
#         datamodule.extra_test={params.extra_test} \
#         """

rule all:
    input:
        data_dir / "clustered_pythia_sig.h5",
        data_dir / "clustered_pythia_bkg.h5",
        data_dir / "clustered_herwig.h5",


rule split_by_labels:
    output:
        data_dir / "clustered_pythia_sig.h5",
        data_dir / "clustered_pythia_bkg.h5",
    input:
        data_dir / "clustered_pythia.h5",
    params:
        data_dir=data_dir,
    resources:
        mem_mb=20_000,
    shell:
        """
        python scripts/split_by_labels.py \
        --input_file=clustered_pythia.h5 \
        --data_dir={params.data_dir} \
        """


rule cluster:
    output:
        data_dir / "clustered_{gen}.h5",
    input:
        data_dir / "{gen}.h5",
    params:
        data_dir=data_dir,
    resources:
        mem_mb=128_000,
        slurm_partition="public-bigmem",
    shell:
        """
        python scripts/cluster.py \
        --input_file={wildcards.gen}.h5 \
        --data_dir={params.data_dir} \
        --n_events=none \
        """


rule reshape_and_sort:
    output:
        data_dir / "{gen}.h5",
    params:
        data_dir=data_dir,
        out_file=lambda w: f"{w.gen}.h5",
        raw_file=lambda w: f"{config['raw_files'][w.gen]}",
    resources:
        mem_mb=128_000,
        slurm_partition="public-bigmem",
    shell:
        """
        python scripts/reshape_and_sort.py \
        --data_dir={params.data_dir} \
        --raw_file={params.raw_file} \
        --out_file={params.out_file} \
        """
