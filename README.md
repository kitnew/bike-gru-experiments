# Hyperparameter Optimization for GRU-based Bike Rental Forecasting

This repository is an open-source implementation of the hyperparameter optimization for GRU-based bike rental forecasting using hybrid approach to search pipeline with grid search, random search and Bayesian optimization.

---

## 📌 Project Goals

The purpose of this project is to implement the hyperparameter optimization for GRU-based bike rental forecasting models not only to find the best hyperparameters but also to compare and test the performance of different search methods. Focuses more on approach to maximize performance rather than on dataset.

---

## 🔧 Features to Implement

- [x] Data preprocessing pipeline
- [x] EDA
- [x] GRU model implementation
- [x] Metrics implementation
- [x] Random search pipeline
- [x] Bayesian search pipeline
- [x] Determine tigher hyperparameters grid
- [x] Grid search pipeline
- [x] Find best hyperparameters
- [x] Train model based on best hyperparameters
- [x] Visualize everything!
- [x] Write report

---

## 📁 Dataset

The model uses the **London Bike Sharing Dataset** dataset:
📦 [Kaggle Dataset](https://www.kaggle.com/datasets/hmavrodiev/london-bike-sharing-dataset)

---

## 🚀 Getting Started

In order to set up the project, you need to walkthrough some steps.

```bash
# Clone the repository
git clone https://github.com/kitnew/bike-gru-experiments.git
cd bike-gru-experiments
```

If you want to just explore results, you can skip to the Explore section.

### Prepare data

```bash
# Install dependencies
pip install -r requirements.txt

# Check config in `config/default.yaml`

# Download the dataset
cd data
./download.sh

# Preprocess the dataset strictly after download
python preprocess.py
```

### Train on your own

If you want to reproduce the results of this project, you can simply use steps that were done by me in `notebooks` directory.
Still, you can run rough modules from `src` directory to test how it works.

```bash
# Go to src directory
cd src

# Train model
python train.py

# OR

# Run search
python search.py
```

In order to change search or training hyperparameters consider modifying source files `main` function.
Feel free to update any of them!

### Explore

Now you can explore experiments, plots, and training results. (:

#### Notebooks

All necessary notebooks are in `notebooks` directory.
Simply open them with any Jupyter notebook viewer.

#### Experiments results

If you would like to look closer at experiments results, you can find them in `experiments/checkpoints` directory. Feel free to use them whatever you want!

#### Plots

Interested in rough visualization of results? You can find them in `experiments/plots` directory.
I'm not really good at matplotlib but I tried to cover most of the important metrics.
---

## 📊 Results & Benchmarks

You can find results in `experiments/plots` directory.
Also you can find a report in `docs/report/report.pdf` written in slovak language.

---

## 📜 License

MIT License