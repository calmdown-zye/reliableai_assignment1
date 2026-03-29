# Assignment 1: Adversarial Attacks on Neural Networks
# (26 Spring) 신뢰할수있는인공지능
# 통계데이터사이언스학과 G202558003 김지혜


## Overview
Implementation of adversarial attack methods (FGSM and PGD) on MNIST and CIFAR-10 datasets.

## Requirements
```bash
pip install -r requirements.txt
```

## How to Run
```bash
python test.py
```

1. Train MNIST classifier (target: ≥95% accuracy)
2. Train CIFAR-10 classifier (target: ≥80% accuracy)
3. Run all 4 attack methods (FGSM targeted/untargeted, PGD targeted/untargeted)
4. Print attack success rates for each method
5. Save visualizations to results/ directory
6. Print ε table (ε ∈ {0.05, 0.1, 0.2, 0.3})

## Dataset
MNIST and CIFAR-10 are automatically downloaded when running test.py.

## Project Structure
```
├── models/
│   ├── mnist_model.py     # MNIST CNN classifier
│   └── cifar_model.py     # CIFAR-10 ResNet18 classifier
├── attacks/
│   ├── fgsm.py            # FGSM targeted & untargeted
│   └── pgd.py             # PGD targeted & untargeted
├── results/               # Saved visualizations (PNG)
├── test.py                # Main script
└── requirements.txt
```

