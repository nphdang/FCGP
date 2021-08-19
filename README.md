# Fair Classification with Gaussian Process
This is the implementation of the FCGP method proposed in the paper "Fairness improvement for black-box classifiers with Gaussian process", Information Sciences journal: https://www.sciencedirect.com/science/article/abs/pii/S0020025521006861

# Introduction
Most machine learning models (classifiers) mainly focus on maximizing the classification accuracy. However, in many real-world applications it is the fairness, not the accuracy, of the classifier that is the crucial factor. People often wish to deploy a machine learning model that offers unbiased predictions in sensitive applications e.g. resume screening or loan approval.

We propose to use Gaussian process (GP) to improve fairness while maintaining high accuracy of the pre-trained classifier. The main idea is that we first train a GP regression to approximate the predictions of the pre-trained classifier. We then adjust the GP regression via its noise variance to achieve two goals: (1) maximize the fairness and (2) minimize the difference between the GP regression and the pre-trained classifier.
Our method is an unsupervised post-processing approach.

## Unsupervised post-processing approach for fair classification
![unsupervised-post-processing](https://github.com/nphdang/FCGP/blob/main/unsupervised_postprocessing.jpg)

## Performance comparision
![performance-comparision](https://github.com/nphdang/FCGP/blob/main/performance_comparison.jpg)

# Installation
1. Python 3.6
2. numpy 1.18
3. scikit-learn 0.23
4. scipy 1.1
5. keras 2.2.4
6. tensorflow 1.10
7. pandas 0.24
8. matplotlib 2.2.3
9. seaborn 0.11

# How to run
1. Train the pre-trained classifier: 1_initial.bat
2. Run the Random baseline: 2_random.bat
3. Run the ROC baseline: 3_roc.bat
4. Run the IGD baseline: 4_igd.bat
5. Run the FCGP-S method: 5_fcgp_s.bat
6. Run the FCGP-L method: 6_fcgp_l.bat
7. Visualize the results: 7_plot_result.bat

# Reference
Dang Nguyen, Sunil Gupta, Santu Rana, Alistair Shilton, Svetha Venkatesh (2021). Fairness Improvement for Black-box Classifiers with Gaussian Process. Information Sciences, 576, 542-556
