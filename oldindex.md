---
title: Clinical Prediction from the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset
---

CS109a, Final Project, Dec, 2017
Group #3: Ning Shen, Lars von Buchholtz

## ADNI Alzheimer's dataset

Alzheimer's disease (AD) is characterized by 2 major diagnostic features: pathological changes in the brain namely beta-amyloid placques and deterioration of the mental state. Neither of these features is sufficient but both are necessary for a definitive AD diagnosis. In this project, we will focus on these 2 features separately and 1.) try to predict CSF beta-amyloid level from gene expression data (mainly by Ning Shen) and 2.) try to predict the mental state from brain imaging data (mainly by Lars von Buchholtz). We discussed this approach with Patrick, and he recommended to include EDA and project statements for both approaches in this (slightly longer) report.

## PART 1: Predicting beta-amyloid levels from gene expression data

The original DREAM challenge failed in predicting beta-amyloid levels given the data provided. Therefore, we decided to include micro-array gene expression data in our attempt to predict beta-amyloid levels. If it is possible to predict beta-amyloid levels in the cerebrospinal fluid (CSF) from gene expression in blood samples, this would provide a much less invasive way to diagnose amyloid placques and the identified signature genes could be used as clinical biomarkers.

**Data cleaning and exploration**

In order to add the gene expression data to the original DREAM challenge data, the gene expression dataframe had to be cleaned and transposed first. It contains information  about gene locus, ~ 49,000 gene expression levels, gene annotation, phase, visit, year of collection etc. This dataframe was then merged by corresponding patient IDs to the DREAM challenge table containing beta-amyloid levels, age, MMSE scores, etc.. This yields combined data for 130 subjects. The outcome variables for this part of the study are the beta-amyloid levels (quantitative) and the SAGE.Q2 class (categorical). SAGE.Q2 is a classification label of amyloid-beta level, where amyloid-beta 42 < 192pg/ul is 1 and amyloid-beta 42 > 192pg/ul is 0.

## PART 2: Predicting mental state (MMSE) from brain imaging data (MRI)

**Data cleaning and exploration**

Since the psychological pathology develops relatively late in the disease progression, it would be advantageous to be able to predict the mental state (MMSE score) from MRI brain imaging data which is routinely acquired in clinical settings. We downloaded the data for this challenge from https://ida.loni.usc.edu/pages/access/studyData.jsp?categoryId=43&subCategoryId=94 
Initial Data Exploration:
The data in the baseline_data.csv file consists of 628 rows corresponding to unique observations of 628 patients. It contains 2150 measurements of brain geometry derived from 3D MRI images, e.g. area, thickness, curvature, etc. of different brain regions. It can be assumed that many of these variables are correlated with each other because of geometrical necessity.
The spreadsheet also contains demographic data such as education, ethnicity, gender, race and age as well as some diagnostic data: MMSE score, Diagnosis, Apoe4 genotype. The Apoe4 allele is a polymorphism of the Apo E gene that is associated with AD susceptibility. In addition, there is an indicator variable for Apoe4 imputation.  
We cleaned up the demographic variables in the following way: gender, ethnicity and genotype imputation were converted to binary variables, race and ApoE4 allele number were converted to binary dummy variables. All ID variables were dropped from the analysis. The remaining variables are all quantitative.



