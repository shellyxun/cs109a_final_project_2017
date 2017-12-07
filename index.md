---
title: Project Overview
notebook: index.ipynb
---

## Contents
{:.no_toc}
*  
{: toc}


Final Project for Course 109A at Harvard Extension School, 2017  

Group #3: Ning Shen, Lars von Buchholtz


## Introduction
  
Alzheimer's disease (AD) is characterized by 2 major diagnostic features: pathological changes in the brain namely beta-amyloid placques and deterioration of the mental state. Neither of these features is sufficient but both are necessary for a definitive AD diagnosis.  
  
In this project, we focused on predicting the mental state in form of the Mini-Mental State Exam (MMSE) score from MRI Imaging Data. This represents a variation of the century old challenge to predict psychology from anatomy ([Cajal, 1899](https://archive.org/details/comparativestud00cajagoog)).  
The MMSE or Folstein test is a 30-point questionnaire that is used extensively in clinical and research settings to measure cognitive impairment and takes about 5 minutes to administer ([Wikipedia article](https://en.wikipedia.org/wiki/Mini%E2%80%93Mental_State_Examination)). A major drawback of the MMSE is that the data is quite noisy in itself and influenced by confounding demographic factors such as age and education. It is however still one of the major tests in the diagnosis of dementia including AD.
Being able to predict the MMSE score from MRI images that are readily available and archived from many patients might help to diagnose potential dementia patients even before cognitive symptoms are detected. It might also help to identify anatomical features that are not confounded by demographic factors and help to diagnose dementia/AD at an early stage. Therefore, it is not surprising that predicting MMSE from MRI data was one of the subchallenges of the [AD Big Data DREAM Challenge](https://www.synapse.org/#!Synapse:syn2290704/wiki/60828). In this project we are closely following the guidelines for this subchallenge.  
The data for this project came from the [Alzheimer's Disease Neuroimaging Initiative (ADNI)](http://adni.loni.usc.edu/) ([Wikipedia](https://en.wikipedia.org/wiki/Alzheimer%27s_Disease_Neuroimaging_Initiative)), a major collaborative effort to collect clinical and biological data related to AD and make it publicly available to researchers.
Major challenges in this prediction task are the noisiness of the MMSE outcome variable itself ([Wikipedia](https://en.wikipedia.org/wiki/Mini%E2%80%93Mental_State_Examination)), the small number of patients in the dataset (628) and the high multi-colinearity in the imaging data ([see EDA](https://shellyxun.github.io/cs109a_final_project_2017/EDA_MMSE.html)).

In a separate side project we explored the possibility of predicting the second feature of AD, beta-amyloid placques, from gene expression profiles in blood samples. This approach, if successful, would have an enormous medical benefit:
Beta-amyloid levels in the cerebrospinal fluid (CSF) and beta-amyloid placques in the brain detected by PET scans are currently the earliest diagnostic manifestations of AD. However, CSF samples are highly invasive and PET scans are semi-invasive and very expensive. It would be very beneficial, if they could be partially replaced by a simple blood test. However, [previous attempts](link) have failed to predict beta-amyloid levels from other patient data and the size of the dataset turned out to be prohibitive for a timely analysis for this subproject. For completeness, we nevertheless include the [highly informative EDA for this part of the project](link) in our report.



## Related Work
 
A [search of the biomedical literature database Pubmed](https://www.ncbi.nlm.nih.gov/pubmed/?term=ADNI) identifies 1007 publications that are at least partially dealing with the ADNI collaborative dataset. [78 of these](https://www.ncbi.nlm.nih.gov/pubmed/?term=ADNI+MMSE) are touching on the subject of the MMSE score in title or abstract showing that there is a lot of scientific interest in this specific topic. A decent number of labs also participated in the [ADNI Big Data DREAM challenge](https://www.synapse.org/#!Synapse:syn2290704/wiki/60828). A leaderboard of the contributions to subchallenge 3 (prediction of MMSE from MRI) can be found [here](https://www.synapse.org/#!Synapse:syn2290704/wiki/68513). Links to short descriptions of the experimental strategies can be found on the same page.



```python

```

