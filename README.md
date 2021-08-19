# FCGP: Fair Classifier with Gaussian Process
This is the implementation of the FCGP method in the paper "Fairness improvement for black-box classifiers with Gaussian process", Information Sciences journal: https://www.sciencedirect.com/science/article/abs/pii/S0020025521006861
# Introduction
Most machine learning models (classifiers) mainly focus on maximizing the classification accuracy. However, in many real-world applications it is the fairness, not the accuracy, of the classifier that is the crucial factor. People often wish to deploy a machine learning model that offers unbiased decisions in sensitive applications e.g. resume screening or loan approval.

We propose to use Gaussian process (GP) to improve fairness while maintaining high accuracy of the pre-trained classifier. The main idea is that we first train a GP regression to approximate the predictions of the pre-trained classifier. We then adjust the GP regression via its noise variance to achieve two goals: (1) maximize the fairness and (2) minimize the difference between the mean function and the pre-trained classifier.

Our method is an unsupervised post-processing approach that is illustrated as below.
