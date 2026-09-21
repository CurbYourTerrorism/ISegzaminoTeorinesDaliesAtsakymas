# Phoneme RBF-SVM

Input: five numeric acoustic features, original ARFF order.
Feature names: V1, V2, V3, V4, V5.

Class 0: nasal; class 1: oral. Threshold: 0.5.
Training/calibration/test: stratified 60/20/20; rng(42); five-fold train CV.
Calibration: sigmoid on independent calibration split.
Test BA: 0.836464; F1: 0.775777; Brier: 0.092166.
Unknown speakers and recording groups; not validated for microphone audio.
Reject nonfinite or incorrectly shaped production inputs. Training NaNs use fold-local medians.
Source code prepared with AI assistance. Results generated only by this MATLAB run.
