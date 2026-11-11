# automatic-ecg-diagnosis (MIT)

Source: https://github.com/antonior92/automatic-ecg-diagnosis at commit 403934867cea25a85b54faca025ac65b1ac18a45,
the code of Ribeiro AH, Ribeiro MH, Paixão GMM, et al. Automatic diagnosis of the 12-lead ECG using a deep neural
network. Nat Commun. 2020;11:1760. doi:10.1038/s41467-020-15432-4. Licence: MIT, Copyright (c) 2020 Antonio Horta
Ribeiro ([LICENSE](LICENSE)).

`lydus-ecg12-dx@0` uses it as follows.

- `src/benchtoward/baselines/lydus_resnet.py` ports the network of `model.py` (`ResidualUnit`, `get_model`) from Keras
  to PyTorch and the optimisation settings of `train.py` (binary cross-entropy, Adam 1e-3, batch 64,
  ReduceLROnPlateau x0.1 after 7 epochs down to 1e-5, early stopping after 9, at most 70 epochs). The model is
  trained from scratch on the LYDUS train split (`benchtoward/lydus-resnet1d`).
- What differs: inputs are the lead set's leads (12, 6, 4, 3 or 2) instead of 12 fixed leads; outputs are the 26
  Challenge 2021 classes instead of six; selection uses BenchToward's validation split instead of a random 2%; the
  weights of the lowest validation loss are kept (the repository saves both best and last).
- The repository's released weights are a reference model and are not used or hosted here.
