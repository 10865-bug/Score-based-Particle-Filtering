# Score-based-Particle-Filtering
This is a repository that draws inspiration from Data Assistance Flow Matching settings and data. We are currently trying to implement it.

---

## Installation

1. Install `uv`:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh

2. Install Python dependencies using `uv`:

   ```bash
   uv sync

3. Activate the Python virtual environment:

   ```bash
   source .venv/bin/activate

4. Test your installation:

   ```bash
   pytest tests

5. Edit the `out_dir` and `run_subdir` fields of the `Conf` class in `src/conf/conf.py` to the directory where you want the output of every experiment to be saved.

---

## Supplementary Documentation

- Hydra: https://hydra.cc/docs/1.3/intro/
- Hydra ORM: https://github.com/reepoi/hydra-orm
- PyTorch: https://pytorch.org/docs/2.4/index.html
- PyTorch Lightning: https://lightning.ai/docs/pytorch/2.5.0/

---

## Running the data assimilation algorithms

All experiments are launched via:

   python src/dafm/main.py dataset=<dataset> model=<model> <other_overrides>...

### Datasets
- _NavierStokes: 2D Navier–Stokes PDE with periodic boundary conditions.
- _KuramotoSivashinsky: 1D chaotic Kuramoto–Sivashinsky PDE.

### Models
- _FlowMatchingMarginal: Flow Matching–based Ensemble Flow Filter (EnFF).
- Other classical baselines (e.g., Bootstrap Particle Filter, EnKF) are also available in the codebase.

### Key Overrides
- model/diffusion_path=PreviousPosteriorToPredictive
- model/inflation_scale=NoScaling
- model/guidance=_Local
- model/guidance/schedule=Constant
- model.guidance.schedule.constant=<value>
- model.diffusion_path.sigma_min=1e-3
- model.sampling_time_step_count=<steps>
- dataset.observation_noise_std=<value>
- dataset.state_dimension=<value>

---

## Examples

Here are two working examples:

   # Run EnFF-F2P for Navier–Stokes
   python src/dafm/main.py dataset=_NavierStokes model=_FlowMatchingMarginal \
     model/diffusion_path=PreviousPosteriorToPredictive \
     model/inflation_scale=NoScaling \
     model/guidance=_Local model/guidance/schedule=Constant \
     model.guidance.schedule.constant=0.001 \
     model.diffusion_path.sigma_min=1e-3 \
     model.sampling_time_step_count=10 \
     dataset.observation_noise_std=0.01

   # Run EnFF-F2P for Kuramoto–Sivashinsky (1024-dim state)
   python src/dafm/main.py dataset=_KuramotoSivashinsky model=_FlowMatchingMarginal \
     model/diffusion_path=PreviousPosteriorToPredictive \
     model/inflation_scale=NoScaling \
     model/guidance=_Local model/guidance/schedule=Constant \
     model.guidance.schedule.constant=0.005 \
     model.diffusion_path.sigma_min=1e-3 \
     model.sampling_time_step_count=5 \
     dataset.state_dimension=1024 \
     dataset.observation_noise_std=0.1

---

## Processing experiment output

We provide Jupyter notebooks in the `notebooks/` directory to process experiment results:

- tune.ipynb
- logged_metrics.ipynb
- sensitivity.ipynb
- classical_comparison.ipynb
- datasets_*.ipynb
- trajectories_*.ipynb

---

## Running experiments in parallel

You can use GNU parallel to run multiple experiments:

   parallel --eta --header : python src/dafm/main.py <override_1>={<param_1>} <override_2>={<param_2>} ... \
     ::: <param_1> <p1value_1> <p1value_2> ... \
     ::: <param_2> <p2value_1> <p2value_2> ...

Note: On network file systems (NFS), SQLite database writes may be corrupted if multiple processes write simultaneously. Use `-j 1` with `-c job` for preflight configuration saving.

---

## References

- Bao, F., Zhang, Z., & Zhang, G. (2024). An ensemble score filter for tracking high-dimensional nonlinear dynamical systems. Computer Methods in Applied Mechanics and Engineering, 432, 117447. DOI: https://www.sciencedirect.com/science/article/pii/S0045782524007023
- Feng, R., Wu, T., Yu, C., Deng, W., & Hu, P. (2025). On the Guidance of Flow Matching. arXiv:2502.02150. http://arxiv.org/abs/2502.02150
