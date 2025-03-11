# Positive and Unlabeled Learning from Hospital Administrative Data: A Novel Approach to Identify Sepsis Cases

This repository contains an implementation of a **two-step positive and unlabeled (PU) learning approach**, inspired by the survey ["Learning from Positive and Unlabeled Data: A Survey"](https://arxiv.org/abs/1811.04820). Our approach is applied to **identify sepsis cases** from hospital administrative data, as detailed in our paper:

> [Application of Positive and Unlabeled Learning: A Novel Approach for Identifying Sepsis Cases from Hospital Administrative Data](https://www.econstor.eu/bitstream/10419/300110/1/wps-2024-02.pdf).

Our implementation builds on the foundational work by Hirokazu Kiyomaru ([@hkiyomaru](https://github.com/hkiyomaru)) and colleagues, available in their collection [PU Learning Algorithms: Implementation Collection](https://github.com/hkiyomaru/pu-learning).

---

## Overview of the Two-Step Approach

### Step 1: Estimation of Reliable Negative Examples via Spy Technique

In the first step, we **identify reliable negative cases** from the unlabeled data using hospital cost attributes. This is done via the **Spy technique**, where a small subset of known positive cases (spies) is mixed into the unlabeled set to estimate which unlabeled cases are likely negative. 

- **Classifier used**: XGBoost
- **Spy share**: 10% of labeled positive cases

These reliable negative examples serve as the foundation for the next classification step. This method accounts for underreported or misclassified sepsis cases commonly found in administrative data.

### Step 2: Training the Final Classifier

In the second step, we **train a binary classifier** using:
- The **positive cases** identified through explicit coding strategies (ICD-10 codes).
- The **reliable negative cases** identified in Step 1.

The classifier learns to distinguish sepsis and non-sepsis cases based on **cost structures**. 

- **Classifier used**: XGBoost

---

## Data

We use **hospital administrative data** from the **Swiss Federal Statistics Office** (years 2017 to 2019). The dataset includes:
- **71 cost attributes** covering direct and indirect costs.
- **Demographic and clinical variables** (e.g., age, gender, ICD-10 diagnosis codes, CHOP procedure codes).

⚠️ **Note**: For PU learning, we **only use cost attributes** to avoid circular reasoning, as diagnosis codes are part of the sepsis label definition.

---

## Applications

Our approach can be used by:
- **Hospitals** to improve the accuracy of administrative data and optimize DRG (Diagnosis-Related Group) assignment.
- **Regulators** to improve DRG cost weights and monitor coding accuracy.
- **Researchers** for more accurate disease surveillance, epidemiology, and health economics studies.

---

## Authors

- Justus Vogel  
- Johannes Cordier ([@johannescordier](https://github.com/johannescordier))

---

## How to Use

1. **Prepare Data**: Extract cost attributes and positive cases based on ICD-10 explicit coding.
2. **Run Step 1**: Identify reliable negative cases using the Spy technique and XGBoost.
3. **Run Step 2**: Train a binary classifier (XGBoost) using positive and reliable negative examples.
4. **Predict**: Apply the classifier to unlabeled cases to estimate sepsis likelihood.
5. **Post-Processing**: Relabel high-probability cases as positive using a threshold (e.g., top 10%, 20%, 30% of predicted scores).

---

## References

- Bekker, J., & Davis, J. (2020). [Learning from Positive and Unlabeled Data: A Survey](https://arxiv.org/abs/1811.04820). *Machine Learning, 109*, 719–760.
- Kiyomaru, H. [PU Learning Algorithms: Implementation Collection](https://github.com/hkiyomaru/pu-learning).

---

## Citation

If you use this repository or methodology in your research, please cite:

> Vogel, J., & Cordier, J. (2024). *Application of Positive and Unlabeled Learning: A Novel Approach for Identifying Sepsis Cases from Hospital Administrative Data*. Working Paper Series, No. 2024-02. [https://www.econstor.eu/bitstream/10419/300110/1/wps-2024-02.pdf](https://www.econstor.eu/bitstream/10419/300110/1/wps-2024-02.pdf).

---

## Acknowledgments

This work builds on and extends the methodological foundations laid by Kiyomaru et al. We are grateful for making their PU learning implementations publicly available to the research community.
