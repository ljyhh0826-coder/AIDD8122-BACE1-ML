# Benchmarking molecular representations and models for BACE-1 inhibitor prediction

Course project for **AIDD8122 – Advanced Topics in Artificial Intelligence & Drug Discovery**
(Macao Polytechnic University, Faculty of Applied Sciences).

β-secretase 1 (BACE-1) is the rate-limiting enzyme of amyloid-β production and a long-standing
Alzheimer's disease target. This project asks a practical question for AI-driven drug discovery:
**how much does the choice of molecular representation, model family and evaluation protocol change
what we conclude about a BACE-1 activity predictor?** Six representations and six model families are
compared against a message-passing graph neural network, under both a random and a Bemis–Murcko
scaffold split, for classification (active / inactive) and regression (pIC50).

## Results at a glance

| | classification (ROC-AUC) | regression (RMSE) |
|---|---|---|
| best representation + model (scaffold split) | ECFP4 + SVM, 0.906 | AtomPair + XGBoost, 0.742 |
| random split, same models | ≈ +0.025 ROC-AUC | ≈ −0.065 RMSE |

Numbers are means over five seeds; see `results/tables/` for the full tables with standard deviations.
The scaffold-split penalty is the headline practical finding: a random split flatters every model,
because its test molecules are much closer to the training set (mean maximum Tanimoto similarity 0.80
versus 0.71).

## Repository layout

```
src/                 all code
  data.py            curation, Bemis-Murcko scaffold split, random split
  features.py        ECFP4, FCFP4, AtomPair, MACCS, RDKit 2D descriptors
  models.py          kNN, logistic/ridge, RF, SVM, XGBoost, MLP (Tanimoto kernels for fingerprints)
  gnn.py             GINE-style message-passing network, written in plain PyTorch
  metrics.py         ROC-AUC, PR-AUC, MCC, F1, balanced accuracy, RMSE, MAE, R2, enrichment factor
  experiment.py      single train/evaluate run plus result-file handling
  run_benchmark.py   experiment 1-2: representation x model x split
  run_hpo.py         experiment 3: random-search hyperparameter optimisation
  run_gnn.py         experiment 4: GNN and its ablations
  run_ensemble.py    experiment 5: validation-selected ensemble
  interpret.py       experiments 6-7: SHAP, atom attribution, applicability domain
  figures_*.py       figures and tables for the report
data/                raw BACE csv, curated csv, cached features, cached split indices
results/             metrics csv files, per-molecule predictions, tables/
figures/             figures used in the report
```

## Reproducing

```bash
pip install -r requirements.txt
python -m src.data            # curate the dataset and build the splits
python -m src.run_benchmark   # ~10 min on 12 CPU threads
python -m src.run_hpo         # random search, the long one
python -m src.run_gnn         # GNN plus ablations
python -m src.run_ensemble
python -m src.interpret
python -m src.figures_data && python -m src.figures_benchmark && python -m src.figures_gnn
```

Every stage caches to disk and skips work that is already done, so an interrupted run can be resumed
by issuing the same command again. All randomness is seeded (`SEEDS = [0, 1, 2, 3, 4]` in
`src/data.py`); re-running reproduces the numbers in the report exactly.

## Data

`data/raw/bace.csv` is the MoleculeNet BACE dataset (1,513 molecules with measured IC50 against human
BACE-1, assembled by Subramanian et al. from ChEMBL and the literature), downloaded from the DeepChem
public S3 bucket. Curation standardises each structure with RDKit (cleanup, parent fragment,
neutralisation), drops duplicates and keeps 1,512 molecules. The classification label follows
MoleculeNet: active if pIC50 ≥ 7 (IC50 ≤ 100 nM).

## Notes on method choices

- **Tanimoto kernels.** Binary fingerprints are compared with Tanimoto similarity rather than the
  Euclidean distance, both in kNN and as a precomputed SVM kernel, which is the standard choice in
  cheminformatics and markedly better here than the default RBF kernel on raw bits.
- **Scaffold split.** Whole Bemis–Murcko scaffold groups are assigned to a single subset, with large
  groups placed first so that they land in the training set, then the rest shuffled per seed. This is
  the balanced randomised variant used by Chemprop, and it gives a seed-to-seed spread that a
  deterministic scaffold split hides.
- **No test-set leakage.** Hyperparameters, the ensemble composition, the GNN early-stopping epoch and
  the per-model best representation are all chosen on the validation split only.

## References

The report (`report/`) cites the full list. The dataset is from Subramanian et al., *J. Chem. Inf.
Model.* 56:1936 (2016); the MoleculeNet benchmark from Wu et al., *Chem. Sci.* 9:513 (2018).
