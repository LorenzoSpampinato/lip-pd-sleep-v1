# Levodopa-Induced Dyskinesia in Parkinson’s Disease (LID-PD) project
This study aims to characterize video polysomnography high-density EEG (vPSG-hdEEG) signals from healthy subjects and patients with Parkinson's disease. In detail, subjects are divided into four groups i.e., *healthy control subjects* (CTL), *de novo patients* (DNV), *advanced fluctuating patients* (ADV), *advanced patients with dyskinesia* (DYS).


## Main objectives
* **1st (*supervised*) task**: patients' classification using disease stage labels i.e., *CTL*, *DNV*, *ADV*, *DYS*,
* **2nd (*unsupervised*) task**: patients' clustering using disease stage labels i.e., *CTL*, *DNV*, *ADV*, *DYS*
* **3rd task**: real-time dyskinesia onset prediction



## Methods
The pipeline for processing hd-EEG signals includes:  
  1. **Hypnogram definition**: the sleep stage analysed are *Awake* (0), *N1* (1), *N2* (2), *N3* (3), *REM* (4).

     > - Set **--run_hypnogram** argument to *True* to run this step (default configuration is *False*)

  2. **Signal pre-processing**  
     * *Channel selection*: 150 scalp-EEG channels are picked and divided per brain region i.e., *Pre-frontal* (Fp), *Frontal* (F), *Central* (C), *Temporal* (T), *Parietal* (P), *Occipital* (O) 
     * *Bad channels interpolation*: spherical spline method is used 
     * *Average re-referencing*  
     * *Trend removal*  
     * *Band-pass filtering* (BPF) [0.1 - 40 Hz]
     * *Artifact Removal* via automatic *Independent Component Analysis* (ICA)
     * *Signal segmentation into 30-s epochs*

  3. **Feature extraction**: three types of features are considered i.e., 45 *time- and frequency- domain single-channel* ('sc') features, 12 *channel-connectivity* ('cc') features, and 4 *network analysis* ('na') features

     > - Pre-processing is performed on recordings to extract features from signals. However, these two steps can be separately managed by setting **--run_preprocess** and **--run_features** arguments (default configurations are *True* and *False*, respectively).  
     > - To analyse just one patient set **--only_class** and **--only_patient** arguments (default configurations are *None*). Options for **--only_class** are *CTL*, *DNV*, *ADV*, *DYS*; and for **--only_patient** is 'PDxxx' (patient id).  
     > - To not run bad channels interpolation set **--run_bad_interpolation** argument to *False* (default configuration is *True*).

  4. **Dataframe generation**: for each patient, a pandas dataframe is defined with fields *'Group'* [CTL, DNV, ADV, or DYS]; *'Subject'* [patient ID i.e., PDxxx]; *'Brain region'* [Fp, F, C, T, P, O]; *'Stage'* [sleep stage for each signal epoch]; and *'Features'* ['sc', 'cc', and 'na' features values]
  
     > - Following feature extraction, for each patient a <ins>numpy matrix</ins> with dimensions [# brain regions, # features, # epochs] is defined. Values from different brain regions are then combined in the Dataframe definition. 
     > - Set **--run_aggregation** argument to *True* to run this step (default configuration is *False*).
     > - Set **--aggregate_labels** argument to *False* to not consider the sleep stages i.e., '\' is used for Dataframe definition (default configuration is *False*).
     
  5. **NaN values identification**: if missing values are found, a figure is generated to show their location within each dataframe

     > - Set **--run_nan_check** argument to *True* to run this step (default configuration is *False*)

  6. **Training, Validation, and Test set definition**: stratify sampling is used

     > - Set **--run_dataset_split** argument to *True* to run this step (default configuration is *False*). 
     > - Optional arguments are **--training_percentage**, **--validation_percentage**, **--test_percentage**, indicating the percentages for splitting the dataset (default configurations are *0.6*, *0.2*, *0.2* respectively)

  7. **Dimensionality reduction** via *Principal Component Analysis* (PCA)

     > - Set **--run_pca** argument to *True* to run this step (default configuration is *False*). 
     > - Optional argument is **--explained_variance**, indicating the percentage of variance to explain the data that consequently determines the number of principal components (default configuration is *0.95*).

  8. **Classification task** via *Random forest* (RF), *Support Vector Machine* (SVM), *Multi-Layer Perceptron* (MLP): for each classifier, an image for representing final results (i.e., Confusion matrix, Precision, Recall, F1-Score) is saved

     > - Set **--run_class_task** argument to *True* to run this step (default configuration is *False*).
     
> [!NOTE]
> Is it possible to execute **Dimensionality reduction** and **Classification task** steps <ins>separately on each sleep stage and/or brain region</ins> by setting **--only_stage** and/or **--only_brain_region** arguments (default configurations are *None*). Options for **--only_stage** are *W*, *N1*, *N2*, *N3*, *REM*; and for **--only_brain_region** are *Fp*, *F*, *C*, *T*, *P*, *O*



## Dataset
### Data folders
Our dataset includes 36 8-hour signals recorded overnight from 256 EEG channels (*GSN-HydroCel-257 montage*).
In order to execute the code successfully, ensure that the following two folders are present:
* *'Dataset'*, which should contain the raw hd-EEG signals (.mff format)
* *'Labels'*, which should contain the associated hypnograms for the hd-EEG signals (.mat format)  

> [!IMPORTANT]
> During code execution, to correctly load the signals the folder of each patient is automatically renamed by adding the *".mff"* extension (e.g., 'PD002' --> 'PD002.mff'). Once processed, original names are restored.

The structure for these two folders must be as follows:
<pre>
├── Dataset           │ ├── Labels  
│ │                   │ │  
│ ├── CTL             │ ├── CTL   
│ │ ├── PD009         │ │ ├── PD009.mat  
│ │ ├── . . .         │ │ ├── . . .  
│ │ └── PD043         │ │ └── PD043.mat  
│ │                   │ │  
│ ├── DNV             │ ├── DNV   
│ │ ├── PD005         │ │ ├── PD005.mat  
│ │ ├── . . .         │ │ ├── . . .  
│ │ └── PD041         │ │ └── PD041.mat  
│ │                   │ │  
│ ├── ADV             │ ├── ADV   
│ │ ├── PD002         │ │ ├── PD002.mat  
│ │ ├── . . .         │ │ ├── . . .  
│ │ └── PD042         │ │ └── PD042.mat  
│ │                   │ │  
│ └── DYS             │ └── DYS   
│   ├── PD004         │   ├── PD004.mat  
│   ├── . . .         │   ├── . . .  
│   └── PD045         │   └── PD045.mat
</pre>


### Data structure
- The <ins>folder of each subject with hd-EEG signals</ins> must contain at least the following:
  * **XML files** i.e., *'coordinates.xml'*, *'epochs.xml'*, *'info.xml'*, *'info1.xml'*, *'info2.xml'*, *'notes.xml'*, *'pnsSet.xml'*, *'po_videoSyncups.xml'*, *'sensorLayout.xml'*, *'subject.xml'*
  * **BIN files** i.e., *'signal1.bin'*, *'signal2.bin'*
    

- For the <ins>hypnograms</ins>: each *.mat* file must show at least the following two fields:
  1. **'sesscoringinfoss'**, containing a numpy matrix where:
     * the 1st and 2nd columns refer to start and end datapoint of each 6-s epochs 
     * the 3rd column represents the sleep stages i.e., Awake (0), N1 (-1), N2 (-2), N3 (-3), REM (1)
     * the 4th column shows the membership index of each 6-second epoch to a 30-second epoch
  2. **'badchannelsNdx'**, containing the list of all the 'bad' hd-EEG channels (list e.g. ['E91', 'E103', 'E205']).



## Usage
The main script to execute is *lidpd_main.py*: all the functions that are used are defined in *lidpd_utils.py*. To run the script, please use the following syntax:

```#RRGGBB
python lidpd_main.py --data_path r'.\Dataset --label_path r'.\Labels --save_path r'.\Results --info_path r'.\Results'
```
