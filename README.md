# IM-ML  
## Injection Moulding - Machine learning
### Machine Learning models applied to Injction Moulding Datasets for produced parts quality prediciton

Used Python 3.12.3  
Needed packages are specified in requirements.txt  

### How to run the code  
- Tune your configuration inside config folder (different config files for different .py files)

  See [test and validation split settings](docs/data_split_configuration.md) for
  configuring the test share and separate HPO and final-training validation shares.

  See [the four-family XGBoost pipeline](docs/xgboost_hybrid_pipeline.md) for
  the pretrained-M1, direct-PP/CNN, and engineered-feature hybrids, PP-only
  Fusion, per-model configuration, and the required physical-time metadata.

  DoE1 defaults use `Clean_Data_1_2_Train_Test_Divided`: the `Train/` data is
  split for fitting, validation and final `test_1`; the complete `Test/` data
  supplies independent `test_2` when configured through `evaluation.test_2_dir`
  in each model's config (set `null` or omit to disable). Fusion SHAP covers the full Train distribution
  and both tests with batch-balanced sampling. Flat datasets retain the single
  test behavior. A new original reference is required for the changed dataset.

- You can run the code with the following command  
```
python src/[name_of_the_file_you_want_to_run].py
```

The available files are:
 - `BC_MLP_IM.py` - Multy layer Perceptron for Binary classification of produced parts (binary label based on wieght)
 - `Reg_MLP_IM.py` -  Multy layer Perceptron for regression, predicting scalar weight of produced parts
