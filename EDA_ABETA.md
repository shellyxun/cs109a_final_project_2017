---
title: EDA of beta-amyloid and gene expression
notebook: EDA_ABETA.ipynb
nav_include: 1
---

## Contents
{:.no_toc}
*  
{: toc}










## Data cleaning and merging
The gene expression data provided in ADNI contains information  about gene locus, ~ 49,000 gene expression levels, gene annotation, phase, visit, year of collection etc. This dataset provides rich information but is not well formated. In order to add the gene expression data to the original DREAM challenge data, the gene expression dataframe had to be cleaned and transposed first. After the cleaning of gene expression data, the converted table merged with the Dream challenge table, generated a well formated table for following EDA. 





    Overview of the gene expression dataset loaded before cleaning.
    





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>738</th>
      <th>739</th>
      <th>740</th>
      <th>741</th>
      <th>742</th>
      <th>743</th>
      <th>744</th>
      <th>745</th>
      <th>746</th>
      <th>747</th>
    </tr>
    <tr>
      <th>0</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Phase</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>ADNIGO</td>
      <td>ADNI2</td>
      <td>ADNI2</td>
      <td>ADNIGO</td>
      <td>ADNI2</td>
      <td>ADNI2</td>
      <td>ADNI2</td>
      <td>ADNIGO</td>
      <td>...</td>
      <td>ADNIGO</td>
      <td>ADNI2</td>
      <td>ADNIGO</td>
      <td>ADNI2</td>
      <td>ADNIGO</td>
      <td>ADNI2</td>
      <td>ADNI2</td>
      <td>ADNI2</td>
      <td>ADNI2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Visit</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>m48</td>
      <td>v03</td>
      <td>v03</td>
      <td>m48</td>
      <td>v03</td>
      <td>v03</td>
      <td>v06</td>
      <td>bl</td>
      <td>...</td>
      <td>bl</td>
      <td>v03</td>
      <td>m60</td>
      <td>v03</td>
      <td>bl</td>
      <td>v03</td>
      <td>v03</td>
      <td>v03</td>
      <td>v06</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>SubjectID</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>116_S_1249</td>
      <td>037_S_4410</td>
      <td>006_S_4153</td>
      <td>116_S_1232</td>
      <td>099_S_4205</td>
      <td>007_S_4467</td>
      <td>128_S_0205</td>
      <td>003_S_2374</td>
      <td>...</td>
      <td>022_S_2379</td>
      <td>014_S_4668</td>
      <td>130_S_0289</td>
      <td>141_S_4456</td>
      <td>009_S_2381</td>
      <td>053_S_4557</td>
      <td>073_S_4300</td>
      <td>041_S_4014</td>
      <td>007_S_0101</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>260/280</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.05</td>
      <td>2.07</td>
      <td>2.04</td>
      <td>2.03</td>
      <td>2.01</td>
      <td>2.05</td>
      <td>1.95</td>
      <td>1.99</td>
      <td>...</td>
      <td>2.05</td>
      <td>2.05</td>
      <td>1.98</td>
      <td>2.09</td>
      <td>1.87</td>
      <td>2.03</td>
      <td>2.11</td>
      <td>1.94</td>
      <td>2.06</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>260/230</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.55</td>
      <td>1.54</td>
      <td>2.1</td>
      <td>1.52</td>
      <td>1.6</td>
      <td>1.91</td>
      <td>1.47</td>
      <td>2.07</td>
      <td>...</td>
      <td>1.9</td>
      <td>2.05</td>
      <td>1.65</td>
      <td>1.56</td>
      <td>1.45</td>
      <td>1.33</td>
      <td>0.27</td>
      <td>1.72</td>
      <td>1.35</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>RIN</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.7</td>
      <td>7.6</td>
      <td>7.2</td>
      <td>6.8</td>
      <td>7.9</td>
      <td>7</td>
      <td>7.9</td>
      <td>7.2</td>
      <td>...</td>
      <td>6.7</td>
      <td>6.5</td>
      <td>6.3</td>
      <td>6.4</td>
      <td>6.6</td>
      <td>6.8</td>
      <td>6.2</td>
      <td>5.8</td>
      <td>6.7</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Affy Plate</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>7</td>
      <td>3</td>
      <td>6</td>
      <td>7</td>
      <td>9</td>
      <td>4</td>
      <td>3</td>
      <td>8</td>
      <td>...</td>
      <td>8</td>
      <td>6</td>
      <td>9</td>
      <td>3</td>
      <td>8</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>YearofCollection</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2011</td>
      <td>2012</td>
      <td>2011</td>
      <td>2011</td>
      <td>2011</td>
      <td>2012</td>
      <td>2011</td>
      <td>2011</td>
      <td>...</td>
      <td>2011</td>
      <td>2012</td>
      <td>2011</td>
      <td>2012</td>
      <td>2011</td>
      <td>2012</td>
      <td>2011</td>
      <td>2011</td>
      <td>2012</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ProbeSet</th>
      <td>LocusLink</td>
      <td>Symbol</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11715100_at</th>
      <td>LOC8355</td>
      <td>HIST1H3G</td>
      <td>2.237</td>
      <td>2.294</td>
      <td>2.14</td>
      <td>2.062</td>
      <td>2.04</td>
      <td>2.439</td>
      <td>1.955</td>
      <td>2.372</td>
      <td>...</td>
      <td>2.34</td>
      <td>2.405</td>
      <td>2.349</td>
      <td>2.212</td>
      <td>2.382</td>
      <td>2.497</td>
      <td>2.309</td>
      <td>2.302</td>
      <td>2.661</td>
      <td>[HIST1H3G] histone cluster 1  H3g</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 747 columns</p>
</div>











    Gene expression dataset glance after cleaning.
    





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Phase</th>
      <th>Visit</th>
      <th>PTID</th>
      <th>260/280</th>
      <th>260/230</th>
      <th>RIN</th>
      <th>Affy Plate</th>
      <th>YearofCollection</th>
      <th>ProbeSet</th>
      <th>11715100_at</th>
      <th>...</th>
      <th>AFFX-r2-TagH_at</th>
      <th>AFFX-r2-TagIN-3_at</th>
      <th>AFFX-r2-TagIN-5_at</th>
      <th>AFFX-r2-TagIN-M_at</th>
      <th>AFFX-r2-TagJ-3_at</th>
      <th>AFFX-r2-TagJ-5_at</th>
      <th>AFFX-r2-TagO-3_at</th>
      <th>AFFX-r2-TagO-5_at</th>
      <th>AFFX-r2-TagQ-3_at</th>
      <th>AFFX-r2-TagQ-5_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>116_S_1249</th>
      <td>ADNIGO</td>
      <td>m48</td>
      <td>116_S_1249</td>
      <td>2.05</td>
      <td>0.55</td>
      <td>7.7</td>
      <td>7</td>
      <td>2011</td>
      <td>NaN</td>
      <td>2.237</td>
      <td>...</td>
      <td>2.355</td>
      <td>2.624</td>
      <td>2.01</td>
      <td>2.906</td>
      <td>2.463</td>
      <td>2.05</td>
      <td>2.06</td>
      <td>1.858</td>
      <td>2.028</td>
      <td>2.162</td>
    </tr>
    <tr>
      <th>037_S_4410</th>
      <td>ADNI2</td>
      <td>v03</td>
      <td>037_S_4410</td>
      <td>2.07</td>
      <td>1.54</td>
      <td>7.6</td>
      <td>3</td>
      <td>2012</td>
      <td>NaN</td>
      <td>2.294</td>
      <td>...</td>
      <td>2.1</td>
      <td>2.82</td>
      <td>1.726</td>
      <td>2.465</td>
      <td>2.26</td>
      <td>1.933</td>
      <td>1.717</td>
      <td>2.208</td>
      <td>2.058</td>
      <td>1.882</td>
    </tr>
    <tr>
      <th>006_S_4153</th>
      <td>ADNI2</td>
      <td>v03</td>
      <td>006_S_4153</td>
      <td>2.04</td>
      <td>2.1</td>
      <td>7.2</td>
      <td>6</td>
      <td>2011</td>
      <td>NaN</td>
      <td>2.14</td>
      <td>...</td>
      <td>2.165</td>
      <td>2.455</td>
      <td>1.84</td>
      <td>2.681</td>
      <td>2.251</td>
      <td>1.985</td>
      <td>1.77</td>
      <td>2.184</td>
      <td>2.007</td>
      <td>2.134</td>
    </tr>
    <tr>
      <th>116_S_1232</th>
      <td>ADNIGO</td>
      <td>m48</td>
      <td>116_S_1232</td>
      <td>2.03</td>
      <td>1.52</td>
      <td>6.8</td>
      <td>7</td>
      <td>2011</td>
      <td>NaN</td>
      <td>2.062</td>
      <td>...</td>
      <td>2.094</td>
      <td>2.599</td>
      <td>1.837</td>
      <td>2.713</td>
      <td>2.158</td>
      <td>1.916</td>
      <td>1.878</td>
      <td>2.163</td>
      <td>2.185</td>
      <td>2.099</td>
    </tr>
    <tr>
      <th>099_S_4205</th>
      <td>ADNI2</td>
      <td>v03</td>
      <td>099_S_4205</td>
      <td>2.01</td>
      <td>1.6</td>
      <td>7.9</td>
      <td>9</td>
      <td>2011</td>
      <td>NaN</td>
      <td>2.04</td>
      <td>...</td>
      <td>1.973</td>
      <td>2.544</td>
      <td>1.909</td>
      <td>2.548</td>
      <td>2.266</td>
      <td>2.077</td>
      <td>1.838</td>
      <td>2.085</td>
      <td>1.941</td>
      <td>1.883</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 49395 columns</p>
</div>







    Gene expression dataset summary after cleaning.
    





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Phase</th>
      <th>Visit</th>
      <th>PTID</th>
      <th>260/280</th>
      <th>260/230</th>
      <th>RIN</th>
      <th>Affy Plate</th>
      <th>YearofCollection</th>
      <th>ProbeSet</th>
      <th>11715100_at</th>
      <th>...</th>
      <th>AFFX-r2-TagH_at</th>
      <th>AFFX-r2-TagIN-3_at</th>
      <th>AFFX-r2-TagIN-5_at</th>
      <th>AFFX-r2-TagIN-M_at</th>
      <th>AFFX-r2-TagJ-3_at</th>
      <th>AFFX-r2-TagJ-5_at</th>
      <th>AFFX-r2-TagO-3_at</th>
      <th>AFFX-r2-TagO-5_at</th>
      <th>AFFX-r2-TagQ-3_at</th>
      <th>AFFX-r2-TagQ-5_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>744</td>
      <td>744</td>
      <td>744</td>
      <td>744</td>
      <td>744</td>
      <td>744</td>
      <td>744</td>
      <td>744</td>
      <td>0.0</td>
      <td>744</td>
      <td>...</td>
      <td>744.000</td>
      <td>744.00</td>
      <td>744.000</td>
      <td>744.000</td>
      <td>744.000</td>
      <td>744.000</td>
      <td>744.000</td>
      <td>744.000</td>
      <td>744.000</td>
      <td>744.000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>12</td>
      <td>744</td>
      <td>40</td>
      <td>173</td>
      <td>38</td>
      <td>9</td>
      <td>4</td>
      <td>0.0</td>
      <td>450</td>
      <td>...</td>
      <td>389.000</td>
      <td>440.00</td>
      <td>374.000</td>
      <td>434.000</td>
      <td>395.000</td>
      <td>360.000</td>
      <td>352.000</td>
      <td>431.000</td>
      <td>382.000</td>
      <td>399.000</td>
    </tr>
    <tr>
      <th>top</th>
      <td>ADNI2</td>
      <td>v03</td>
      <td>116_S_4209</td>
      <td>2.03</td>
      <td>1.65</td>
      <td>7.1</td>
      <td>3</td>
      <td>2011</td>
      <td>NaN</td>
      <td>2.428</td>
      <td>...</td>
      <td>2.181</td>
      <td>2.71</td>
      <td>1.828</td>
      <td>2.613</td>
      <td>2.337</td>
      <td>1.983</td>
      <td>1.849</td>
      <td>2.262</td>
      <td>1.925</td>
      <td>2.125</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>449</td>
      <td>357</td>
      <td>1</td>
      <td>73</td>
      <td>13</td>
      <td>61</td>
      <td>88</td>
      <td>383</td>
      <td>NaN</td>
      <td>5</td>
      <td>...</td>
      <td>7.000</td>
      <td>7.00</td>
      <td>7.000</td>
      <td>6.000</td>
      <td>12.000</td>
      <td>9.000</td>
      <td>9.000</td>
      <td>6.000</td>
      <td>7.000</td>
      <td>7.000</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 49395 columns</p>
</div>







    Dream data 2nd question training set glance.
    





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RID</th>
      <th>PTID</th>
      <th>AGE</th>
      <th>PTGENDER</th>
      <th>PTEDUCAT</th>
      <th>APOE4</th>
      <th>MMSE</th>
      <th>ABETA</th>
      <th>SAGE.Q2</th>
      <th>APOE Genotype</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>011_S_0005</td>
      <td>73.7</td>
      <td>Male</td>
      <td>16</td>
      <td>0</td>
      <td>29</td>
      <td>115.0</td>
      <td>1</td>
      <td>3,3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>19</td>
      <td>067_S_0019</td>
      <td>73.1</td>
      <td>Female</td>
      <td>18</td>
      <td>0</td>
      <td>29</td>
      <td>260.0</td>
      <td>0</td>
      <td>2,3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>31</td>
      <td>023_S_0031</td>
      <td>77.7</td>
      <td>Female</td>
      <td>18</td>
      <td>0</td>
      <td>30</td>
      <td>240.0</td>
      <td>0</td>
      <td>3,3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>43</td>
      <td>018_S_0043</td>
      <td>76.2</td>
      <td>Male</td>
      <td>16</td>
      <td>0</td>
      <td>29</td>
      <td>175.0</td>
      <td>1</td>
      <td>2,3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>47</td>
      <td>100_S_0047</td>
      <td>84.7</td>
      <td>Male</td>
      <td>20</td>
      <td>0</td>
      <td>30</td>
      <td>252.0</td>
      <td>0</td>
      <td>2,3</td>
    </tr>
  </tbody>
</table>
</div>







    Dream data 2nd question training set summary.
    





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RID</th>
      <th>AGE</th>
      <th>PTEDUCAT</th>
      <th>APOE4</th>
      <th>MMSE</th>
      <th>ABETA</th>
      <th>SAGE.Q2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>176.000000</td>
      <td>176.000000</td>
      <td>176.000000</td>
      <td>176.000000</td>
      <td>176.000000</td>
      <td>176.000000</td>
      <td>176.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2543.375000</td>
      <td>75.309091</td>
      <td>16.255682</td>
      <td>0.272727</td>
      <td>29.028409</td>
      <td>195.500000</td>
      <td>0.460227</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1874.399901</td>
      <td>5.364692</td>
      <td>2.600543</td>
      <td>0.517737</td>
      <td>1.239488</td>
      <td>53.370559</td>
      <td>0.499838</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5.000000</td>
      <td>62.000000</td>
      <td>8.000000</td>
      <td>0.000000</td>
      <td>24.000000</td>
      <td>75.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>558.000000</td>
      <td>71.775000</td>
      <td>14.000000</td>
      <td>0.000000</td>
      <td>29.000000</td>
      <td>149.950000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4032.500000</td>
      <td>74.900000</td>
      <td>16.000000</td>
      <td>0.000000</td>
      <td>29.000000</td>
      <td>202.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4281.250000</td>
      <td>78.400000</td>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>30.000000</td>
      <td>240.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4516.000000</td>
      <td>89.600000</td>
      <td>20.000000</td>
      <td>2.000000</td>
      <td>30.000000</td>
      <td>302.800000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>










<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RID</th>
      <th>PTID</th>
      <th>AGE</th>
      <th>PTGENDER</th>
      <th>PTEDUCAT</th>
      <th>APOE4</th>
      <th>MMSE</th>
      <th>ABETA</th>
      <th>SAGE.Q2</th>
      <th>APOE Genotype</th>
      <th>...</th>
      <th>AFFX-r2-TagH_at</th>
      <th>AFFX-r2-TagIN-3_at</th>
      <th>AFFX-r2-TagIN-5_at</th>
      <th>AFFX-r2-TagIN-M_at</th>
      <th>AFFX-r2-TagJ-3_at</th>
      <th>AFFX-r2-TagJ-5_at</th>
      <th>AFFX-r2-TagO-3_at</th>
      <th>AFFX-r2-TagO-5_at</th>
      <th>AFFX-r2-TagQ-3_at</th>
      <th>AFFX-r2-TagQ-5_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>40</th>
      <td>984</td>
      <td>021_S_0984</td>
      <td>76.6</td>
      <td>Male</td>
      <td>14</td>
      <td>1</td>
      <td>30</td>
      <td>75.0</td>
      <td>1</td>
      <td>3,4</td>
      <td>...</td>
      <td>2.258</td>
      <td>3.229</td>
      <td>1.817</td>
      <td>2.565</td>
      <td>2.13</td>
      <td>1.829</td>
      <td>1.596</td>
      <td>2.394</td>
      <td>1.887</td>
      <td>2.09</td>
    </tr>
    <tr>
      <th>77</th>
      <td>4179</td>
      <td>033_S_4179</td>
      <td>83.0</td>
      <td>Male</td>
      <td>20</td>
      <td>2</td>
      <td>30</td>
      <td>82.7</td>
      <td>1</td>
      <td>4,4</td>
      <td>...</td>
      <td>2.196</td>
      <td>2.655</td>
      <td>1.883</td>
      <td>2.46</td>
      <td>2.24</td>
      <td>2.002</td>
      <td>2.052</td>
      <td>2.262</td>
      <td>1.982</td>
      <td>2.024</td>
    </tr>
    <tr>
      <th>95</th>
      <td>4339</td>
      <td>082_S_4339</td>
      <td>84.3</td>
      <td>Male</td>
      <td>17</td>
      <td>2</td>
      <td>29</td>
      <td>90.7</td>
      <td>1</td>
      <td>4,4</td>
      <td>...</td>
      <td>2.274</td>
      <td>2.747</td>
      <td>1.918</td>
      <td>2.67</td>
      <td>2.402</td>
      <td>1.792</td>
      <td>1.808</td>
      <td>2.588</td>
      <td>2.013</td>
      <td>2.304</td>
    </tr>
    <tr>
      <th>124</th>
      <td>4474</td>
      <td>031_S_4474</td>
      <td>85.6</td>
      <td>Male</td>
      <td>18</td>
      <td>0</td>
      <td>28</td>
      <td>92.5</td>
      <td>1</td>
      <td>3,3</td>
      <td>...</td>
      <td>2.178</td>
      <td>2.941</td>
      <td>1.878</td>
      <td>3.114</td>
      <td>2.421</td>
      <td>2.028</td>
      <td>1.76</td>
      <td>2.253</td>
      <td>1.94</td>
      <td>2.089</td>
    </tr>
    <tr>
      <th>94</th>
      <td>4335</td>
      <td>021_S_4335</td>
      <td>71.7</td>
      <td>Female</td>
      <td>15</td>
      <td>0</td>
      <td>30</td>
      <td>95.4</td>
      <td>1</td>
      <td>3,3</td>
      <td>...</td>
      <td>2.342</td>
      <td>2.579</td>
      <td>1.905</td>
      <td>2.725</td>
      <td>2.472</td>
      <td>1.804</td>
      <td>2.122</td>
      <td>2.035</td>
      <td>1.932</td>
      <td>1.942</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 49404 columns</p>
</div>







    The dimension for the new merged table is:
    





    (130, 49404)



## EDA

### 1) After data cleaning and merging, the first question we ask is: is amyloid beta level distribution bimodal in the data we work with? In addition, are the two classes that we are going to classify balanced? 

Fig1 addresses this question, showing bimodal distribution in both the original dream data and the merged table and well-balanced data .






![png](EDA_ABETA_files/EDA_ABETA_12_0.png)


### 2) Is there any correlation/difference in the two classes with regard to age, apo4 level, MMSE score, and year of collection? 

Fig2 addresses these questions. The Age of patients in class 0 are more concentrated around 75 compared to class 1 (Fig2a). The year of collection of patient samples in class 0 are more spread out than in class 1 (Fig2d). The Apoe4 level in class 0 is much more concentrated near 0, whereas class 1 shows more variation in both directions (Fig2b). When comparing Apoe4 level with ABETA level, there is a clear negative correlation between the two (Fig2f). The MMSE score distribution in class 0 is lower than in class 1 (Fig2c). The scatterplot comparing ABETA level vs MMSE score (Fig2e) showed similar trend, implying high levels of beta-amyloid might correlate with lower cognition. 







![png](EDA_ABETA_files/EDA_ABETA_14_0.png)


### 3) How does high level look of gene expression profile differ between the two classes? 

To address this question, we made a heatmap plot with each row correspond to a patient sample with class label on the side, each column correspond to a gene (We included 100 random genes in the plot for illustration purposes). If the gene expression profile of the two classes are distinctly different, we would expect the heatmap clustering on the y-axis (class bar on the left side of the heatmap) be very well separated between the two classes (red vs blue). The heatmap suggests the gene expression profiles between the two classes are not distinctly different based on this limited subset of genes. However, some of the samples do cluster close, showing thick red/blue class bars on the left, suggesting there might be potential that gene expression profile can help separate the two classes.






![png](EDA_ABETA_files/EDA_ABETA_16_0.png)


## Summary

The basic data cleaning and EDA suggested gene expression profile might be able to help classifying the ABETA groups. In order to confirm this, further feature selection and dimension reduction like PCA need to be applied for modeling. For the modeling part, this can be pursued as either classification or regression. However, due the limitation of time and effort, we decide to focus the modeling the part 2 of the project - predicting mental scores from MRI image data. 
