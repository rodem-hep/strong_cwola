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
    gen="herwig|pythia",
    fold="[0-9]+",
    seed="[0-9]+",
    mode="|un_pretrained_",  # The | is a hack to allow an empty string


# Important paths become Paths
data_dir = Path(config["data_dir"])
project_name = config["project_name"]
folds = range(config["num_folds"])
seeds = range(config["num_seeds"])

# Generate the gen_dope combinations (Herwig has no dope)
gen_dope = [f"pythia_{dope}" for dope in config["dope"]] + ["herwig_0"]

########################################

rule all:
    input:
        expand(data_dir / f"{project_name}/{{mode}}sic.pdf", mode=["", "un_pretrained_"]),
        # expand(data_dir / f"{project_name}/{{mode}}sic.pdf", mode=["un_pretrained_"]),
        data_dir / f"{project_name}/low_level_data.pdf",
        data_dir / f"{project_name}/bdt_sic.pdf"

rule plot_sic:
    input:
        expand(
            data_dir / "{{project_name}}/{{mode}}{gen_dope}_seed_{seed}_combined/{file}.h5",
            gen_dope=gen_dope,
            seed=seeds,
            file=["pythia", "herwig"],
        ),
        "scripts/plot_sic.py",
    output:
        data_dir / "{project_name}/{mode}sic.pdf"
    params:
        data_dir = data_dir / f"{project_name}",
        pretrained = lambda w: "" if w.mode == "un_pretrained_" else "--pretrained",
    resources:
        runtime=5,
    shell:
        "python scripts/plot_sic.py --data_dir={params.data_dir} --output={output} {params.pretrained}"


rule combine_folds:
    """Combine the test sets from each fold into a single file."""
    input:
        expand(data_dir / "{{project_name}}/{{mode}}{{gen_dope}}_seed_{{seed}}_fold_{fold}/original_test.h5", fold=folds),
        expand(data_dir / "{{project_name}}/{{mode}}{{gen_dope}}_seed_{{seed}}_fold_{fold}/additional_test.h5", fold=folds),
    output:
        data_dir / "{project_name}/{mode}{gen_dope}_seed_{seed}_combined/pythia.h5",
        data_dir / "{project_name}/{mode}{gen_dope}_seed_{seed}_combined/herwig.h5",
    params:
        data_dir = lambda w: data_dir / f"{w.project_name}",
        pattern = lambda w: f"{w.mode}{w.gen_dope}_seed_{w.seed}_fold_*",
        output_path = lambda w: f"{w.mode}{w.gen_dope}_seed_{w.seed}_combined",
    shell:
        """
        python scripts/combine_folds.py \
        --data_dir={params.data_dir} \
        --pattern={params.pattern} \
        --output_path={params.output_path} \
        """


rule train_and_save_predictions:
    """For each seed, fold, dope, etc, train a classifier and export the scores."""
    input:
        data_dir / "clustered_pythia_sig.h5",
        lambda w: data_dir / ("clustered_herwig_bkg.h5" if w.gen == "herwig" else "clustered_pythia_bkg.h5"),
        data_dir / "{project_name}/ssfm/done.txt",
    output:
        data_dir / "{project_name}/{mode}{gen}_{dope}_seed_{seed}_fold_{fold}/original_test.h5",
        data_dir / "{project_name}/{mode}{gen}_{dope}_seed_{seed}_fold_{fold}/additional_test.h5",
    params:
        output_dir=data_dir,
        n_sig=90_000,
        n_bkg=1000_000,
        num_folds=config["num_folds"],
        network_name=lambda w: f"{w.mode}{w.gen}_{w.dope}_seed_{w.seed}_fold_{w.fold}",
        bkg_file=lambda w: "clustered_herwig_bkg.h5" if w.gen == "herwig" else "clustered_pythia_bkg.h5",
        extra_test=lambda w: "clustered_pythia_bkg.h5" if w.gen == "herwig" else "clustered_herwig_bkg.h5",
        backbone_path = lambda w: "" if w.mode == "un_pretrained_" else f"model.backbone_path={data_dir}/{project_name}/ssfm/backbone.pkl"
    resources:
        slurm_partition="shared-gpu,private-dpnc-gpu",
        runtime=60 * 12,
        cpus_per_task=6,
        slurm_extra="--gres=gpu:1 --constraint=COMPUTE_TYPE_AMPERE",
    shell:
        """
        python scripts/train.py \
        network_name={params.network_name} \
        output_dir={params.output_dir} \
        project_name={wildcards.project_name} \
        seed={wildcards.seed} \
        datamodule.n_sig={params.n_sig} \
        datamodule.n_bkg={params.n_bkg} \
        datamodule.n_dope={wildcards.dope} \
        datamodule.test_fold={wildcards.fold} \
        datamodule.num_folds={params.num_folds} \
        datamodule.bkg_file={params.bkg_file} \
        datamodule.sig_file=clustered_pythia_sig.h5 \
        datamodule.extra_test={params.extra_test} \
        {params.backbone_path} \
        """


rule pretrain:
    """Create a pretrained model using SSFM."""
    input:
        data_dir / "clustered_pythia_sig.h5",
        data_dir / "clustered_pythia_bkg.h5",
        data_dir / "clustered_herwig_bkg.h5",
    output:
        data_dir / "{project_name}/ssfm/backbone.pkl",
        data_dir / "{project_name}/ssfm/done.txt",
    params:
        output_dir=data_dir,
    resources:
        slurm_partition="shared-gpu,private-dpnc-gpu",
        runtime=60 * 8,
        cpus_per_task=6,
        mem_mb=40_000, # Lots of memory as the whole Mjj spectrum is loaded
        slurm_extra="--gres=gpu:1 --constraint=COMPUTE_TYPE_AMPERE",
    shell:
        """
        python scripts/train.py \
        experiment=pretrain \
        network_name=ssfm \
        output_dir={params.output_dir} \
        project_name={wildcards.project_name} \
        """


rule plot_sic_bdt:
    input:
        expand(
            data_dir / project_name / "bdt_{gen_dope}_seed_{seed}/{file}.h5",
            gen_dope=gen_dope,
            seed=seeds,
            file=["pythia", "herwig"],
        ),
        "scripts/plot_sic.py",
    output:
        data_dir / f"{project_name}/bdt_sic.pdf"
    params:
        data_dir = data_dir / f"{project_name}",
    resources:
        runtime=5,
    shell:
        "python scripts/plot_sic.py --data_dir={params.data_dir} --output={output} --is_bdt --pattern=bdt_*"



rule train_bdts:
    """Train BDTs on the high level features."""
    input:
        data_dir / "clustered_pythia_bkg.h5",
        data_dir / "clustered_herwig_bkg.h5",
        signal = data_dir / "clustered_pythia_sig.h5",
        script = "scripts/train_bdt.py",
    output:
        data_dir / project_name / "bdt_{gen}_{dope}_seed_{seed}/pythia.h5",
        data_dir / project_name / "bdt_{gen}_{dope}_seed_{seed}/herwig.h5",
    params:
        n_sig=90_000,
        n_bkg=1000_000,
        num_folds=config["num_folds"],
        bkg_file=lambda w: f"{data_dir}/clustered_herwig_bkg.h5" if w.gen == "herwig" else f"{data_dir}/clustered_pythia_bkg.h5",
        extra_test=lambda w: f"{data_dir}/clustered_pythia_bkg.h5" if w.gen == "herwig" else f"{data_dir}/clustered_herwig_bkg.h5",
        output_dir = lambda w: data_dir / project_name / f"bdt_{w.gen}_{w.dope}_seed_{w.seed}",
        is_herwig=lambda w: "--is_herwig" if w.gen == "herwig" else "",
    resources:
        mem_mb=30_000,
    shell:
        """
        python scripts/train_bdt.py \
        --background {params.bkg_file} \
        --signal {input.signal} \
        --add_test {params.extra_test} \
        --seed {wildcards.seed} \
        --n_dope {wildcards.dope} \
        --num_folds {params.num_folds} \
        --n_sig {params.n_sig} \
        --n_bkg {params.n_bkg} \
        --num_ensemble 50 \
        --output_dir {params.output_dir} \
        {params.is_herwig}
        """


rule plot_data:
    """Plot the data for the given gen."""
    input:
        signal = data_dir / "clustered_pythia_sig.h5",
        bkg = data_dir / "clustered_pythia_bkg.h5",
        simulation = data_dir / "clustered_herwig_bkg.h5",
        script = "scripts/compare_data.py",
    output:
        data_dir / "{project_name}/low_level_data.pdf",
    shell:
        """
        python {input.script} \
        --signal={input.signal} \
        --bkg={input.bkg} \
        --simulation={input.simulation} \
        --output={output}
        """


rule cluster:
    """Cluster the events and extract the leading two jets and their constituents."""
    input:
        data_dir / "{gen}_{sorb}.h5",
    output:
        data_dir / "clustered_{gen}_{sorb}.h5",
    params:
        data_dir=data_dir,
        event_id_start=lambda w: config["event_id_start"][f"{w.gen}_{w.sorb}"],
    resources:
        mem_mb=200_000,  # Ridiculous amount of memory - need to fix!
        slurm_partition="public-bigmem,shared-bigmem",
    shell:
        """
        python scripts/cluster.py \
        --input_file={wildcards.gen}_{wildcards.sorb}.h5 \
        --data_dir={params.data_dir} \
        --n_events=none \
        --event_id_start={params.event_id_start} \
        """


rule sort_split:
    """Reshape the constituents, sort each event by pT and split into sig and bkg."""
    output:
        data_dir / "{gen}_bkg.h5",
        data_dir / "{gen}_sig.h5",
    params:
        data_dir=data_dir,
        raw_file=lambda w: f"{config['raw_files'][w.gen]}",
    resources:
        mem_mb=100_000,
        slurm_partition="public-bigmem,shared-bigmem",
    shell:
        """
        python scripts/sort_split.py \
        --data_dir={params.data_dir} \
        --raw_file={params.raw_file} \
        --out_flag={wildcards.gen} \
        """
