## SBIsmfs: Simulation-based inference for single-molecule force spectroscopy
SBIsmfs is a Python module for simulation-based inference for single-molecule force spectroscopy experiments. The module provides tools for simulating syntehtic force-spectrscopy experiments, training an amortized and sequential posterior model, and sampling from the posterior distribution. The code is based on the [sbi-toolkit](https://sbi-dev.github.io/sbi/latest/), which is using Pytorch.

![](figure.png)


## Installing code 
Clone the code repository:
```commandline
git clone https://github.com/Dingel321/SBIsmfs.git 
cd SBIsmfs
```

Create a Conda environment with the required dependencies:
```commandline
conda create --name sbi-smfs python=3.9
conda activate sbi-smfs
pip install -r requirements.txt
```

Install with
```commandline
python3 -m pip install .
```

## Running a simulation
```python
    from sbi_smfs.simulator import get_simulator_from_config

    simulator = get_simulator_from_config(
        config_file='path_to_config_file.config', 
        return_q=True
    )

    parameters = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])  # Example parameters

    # Simulate a single trajectory
    q = simulator(parameters)
```

## Running a simple inference
The code can be run with the following command:
```commandline
    train_sequential_posterior \
        --config_file "path_to_config_file.config" \
        --num_rounds number_of_sbi_rounds \
        --num_sim_per_round number_of_simulations_per_round \
        --num_workers number_of_workers \
        --observation_file "file_to_smfs_trajectory" \
        --posterior_file "name_for_posterior_model" \
        --device device \
        --save_interval Save_interval_for_the_posterior_model
```
Simple visualization of the posterior distribtuion:
```python
    import pickle
    import torch
    from sbi_smfs.analysis.plot_posterior import plot_spline_ensemble

    with open(f"path_to_posterior", "rb") as handle:
        posterior = pickle.load(handle)

    samples = posterior.sample((1000000,))
    samples[:, 2:] = samples[:, 2:] - torch.mean(samples[:, 2:], dim=1).reshape(-1, 1)

    plot_spline_ensemble(samples, 1000, config_file, label='Samples', color="blue", alpha=0.02)
```

## Minimal config file example
```config
    [SIMULATOR]
    dt = 0.01
    num_steps = 100000000
    num_knots = 10
    saving_freq = 100
    min_x = 0
    Dx = 0.38
    max_x = 50
    max_G_0 = 30
    max_G_1 = 10
    init_xq_range=10, 40

    [PRIORS]
    type_spline=GAUSSIAN
    parameters_spline=0, 4
    norm_spline_nodes=yes
    type_Dq=UNIFORM
    parameters_Dq=-1, 1
    type_k=UNIFORM
    parameters_k=-2, 0

    [SUMMARY_STATS]
    min_bin = 10
    max_bin = 40
    num_bins = 15
    lag_times = 1, 10, 100, 1000, 10000, 100000
```

## Installation can be tested with:
Install pytest with `pip install pytest` 

Run tests with `pytest tests/`
