# IM-ML  
## Injection Moulding - Machine learning
### Machine Learning models applied to Injction Moulding Datasets for produced parts quality prediciton

Used Python 3.12.3  
Needed packages are specified in requirements.txt  

DOE1 supports batch-average/deviation weight prediction, beta sweeps, and
three loss variants across M1, M2, and Fusion. See
[configuration and output details](docs/weight_decomposition.md).

### How to run the code  
- Tune your configuration inside config folder (different config files for different .py files)

- You can run the code with the following command  
```
python src/[name_of_the_file_you_want_to_run].py
```

The available files are:
 - `BC_MLP_IM.py` - Multy layer Perceptron for Binary classification of produced parts (binary label based on wieght)
 - `Reg_MLP_IM.py` -  Multy layer Perceptron for regression, predicting scalar weight of produced parts
