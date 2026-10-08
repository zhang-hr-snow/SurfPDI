# SurfPDI

SurfPDI predicts protein-DNA binding sites using molecular surface and residue
graphs, Cross-Modal Surface-Residue Attention (CSRA), and a Transformer encoder.

## Installation

Use Python 3.10 or later. Install PyTorch, torch-geometric, torch-scatter, and
torch-cluster with versions matching your CUDA runtime, then:

```bash
pip install -r requirements.txt
```

For molecular surface generation, also install:

```bash
pip install -r requirements-surface.txt
```

## Data preparation

Original benchmark sequences and binding labels are in `data/raw/txt/`.
Place matching protein chain structures at
`data/raw/pdb/<split>/<protein_id>.pdb`, where `<split>` is `train573`,
`test129`, or `test181`.

Packing requires per-residue 2560-D ESMC features. Place them at
`<embedding-cache>/npy573/<protein_id>/esmc_features.pt`, using `npy129` and
`npy181` for the test sets. Run from the repository root:

```bash
python scripts/preprocess/00_build_manifest.py
python scripts/preprocess/01_extract_pdb_foundation.py
python scripts/docker_surface/02_generate_surface_geometry.py --resolution 1.0
python scripts/preprocess/03_map_surface_features.py --surface-version resolution_1
python scripts/preprocess/04_sample_surface_points.py --surface-version resolution_1 --method full --output-version res1.0_full
python scripts/preprocess/05_build_surface_graph.py --sample-version res1.0_full --radius 4.5
python scripts/preprocess/06_build_residue_graph.py --radius 10.0
python scripts/preprocess/07_pack_samples.py --processed-version base_res1.0_full --sample-version res1.0_full --surface-graph-version res1.0_full_r4.5 --residue-graph-version r10 --old-project-root /path/to/embedding-cache
python scripts/preprocess/08_check_processed_samples.py --processed-root data/processed/base_res1.0_full
```

Surface generation requires a working PyKeOps/CUDA environment. Structure
prediction and ESMC feature extraction are separate input preparation steps.

## Training

The only configuration file is `configs/final.yaml`. It contains the final
pretraining and supervised settings; each entry reads its corresponding stage.

```bash
python experiments/pretrain_multimodal_full/run_pretrain_cv.py --config configs/final.yaml --output-dir results/pretraining/paper_full
python scripts/train/train_cv.py --config configs/final.yaml --pretrain-run results/pretraining/paper_full --output-root results/training --run-name paper_full
```

The supervised run performs five-fold training and external benchmark
evaluation. Outputs are saved under `results/training/`.
The reported five-fold protein IDs are in `docs/splits/`.
