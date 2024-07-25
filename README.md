# HandcraftedML
In order to better understand different machine learning & deep learning models, I try to implement models from scratch. **Complex models might directly use pytorch implementation of simple functions** (For example, this repo will not repeated implement or reference linear layer in `PyTorch`).

## Traditional Machine Learning Models
Try to implement all traditional machine learning models with consistent format, now the models follow `.fit()` and `.predict()` functions to train and predict like `SKLearn`.

## Deep Learning Models
For deep learning models, I use `PyTorch` to avoid the calculation of back-prop and derivatives. All implemented models are directly to use like any other `PyTorch` models.


## Dependency
- `PyTorch`
- `NumPy`