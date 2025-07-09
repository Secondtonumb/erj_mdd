# JSON Dataset Analysis Summary

## Overview
This analysis examines two JSON files containing speech recognition data:
- **Train Dataset**: `train_erj_spk_open_train-train.json` (511 entries)
- **Dev Dataset**: `train_erj_spk_open_train-dev.json` (119 entries)

## Speaker Distribution

### Train Dataset Speakers (17 total) - **NOT BALANCED**
- **F02**: 58 utterances (11.4%) - Most represented
- **F03**: 52 utterances (10.2%)
- **M01**: 55 utterances (10.8%)
- **M03**: 53 utterances (10.4%)
- **M02**: 49 utterances (9.6%)
- **M04**: 47 utterances (9.2%)
- **F05**: 45 utterances (8.8%)
- **F04**: 43 utterances (8.4%)
- **M05**: 39 utterances (7.6%)
- **F06**: 36 utterances (7.0%)
- **F08**: 7 utterances (1.4%)
- **M08**: 8 utterances (1.6%)
- **F09**: 4 utterances (0.8%)
- **M09**: 4 utterances (0.8%)
- **M10**: 4 utterances (0.8%)
- **M11**: 4 utterances (0.8%)
- **F10**: 3 utterances (0.6%) - Least represented

**Balance Statistics:**
- Min utterances per speaker: 3
- Max utterances per speaker: 58
- Mean utterances per speaker: 30.1
- Standard deviation: 21.7
- Coefficient of variation: 72.3%
- **Status: NOT BALANCED** (max difference = 55 utterances)

### Dev Dataset Speakers (14 total) - **NOT BALANCED**
- **F02**: 14 utterances (11.8%)
- **F03**: 13 utterances (10.9%)
- **M01**: 13 utterances (10.9%)
- **M03**: 13 utterances (10.9%)
- **M02**: 12 utterances (10.1%)
- **F05**: 11 utterances (9.2%)
- **M04**: 11 utterances (9.2%)
- **F04**: 10 utterances (8.4%)
- **F06**: 9 utterances (7.6%)
- **M05**: 9 utterances (7.6%)
- **F08**: 1 utterance (0.8%)
- **F09**: 1 utterance (0.8%)
- **M08**: 1 utterance (0.8%)
- **M10**: 1 utterance (0.8%)

**Balance Statistics:**
- Min utterances per speaker: 1
- Max utterances per speaker: 14
- Mean utterances per speaker: 8.5
- Standard deviation: 5.0
- Coefficient of variation: 58.3%
- **Status: NOT BALANCED** (max difference = 13 utterances)

**Key Observations:**
- Both datasets have **severe speaker imbalance**
- The most represented speaker (F02) has 19x more utterances than the least represented (F10) in train set
- Same imbalance pattern exists in both datasets
- 3 speakers (F10, M09, M11) are in train but not in dev
- Train-to-dev ratio is approximately 4:1 for most common speakers

## Duplicate Sentences Analysis

### Train Dataset
- **163 unique sentences** appear multiple times (all appear exactly 2 times)
- This represents a significant amount of duplication within the training data
- Each duplicate sentence is spoken by different speakers, suggesting intentional cross-speaker sentence repetition

### Dev Dataset
- **7 unique sentences** appear multiple times (all appear exactly 2 times)
- Much lower duplication rate compared to train dataset
- Similar pattern of different speakers saying the same sentence

### Cross-Dataset Overlap
- **68 sentences** appear in both train and dev datasets
- This represents 19.5% of train sentences and 60.7% of dev sentences
- Significant overlap between datasets, which could impact model evaluation

## Duration Analysis

### Train Dataset
- **Mean duration**: 4.477s
- **Median duration**: 4.262s
- **Range**: 2.018s - 13.182s
- **Distribution**:
  - 2-4s: 38.7% (198 utterances)
  - 4-6s: 51.5% (263 utterances)
  - 6-8s: 7.6% (39 utterances)
  - 8+s: 2.2% (11 utterances)

### Dev Dataset
- **Mean duration**: 4.251s
- **Median duration**: 4.001s
- **Range**: 2.218s - 7.730s
- **Distribution**:
  - 2-4s: 48.7% (58 utterances)
  - 4-6s: 42.0% (50 utterances)
  - 6-8s: 9.2% (11 utterances)
  - 8+s: 0.0% (0 utterances)

## Key Findings and Recommendations

### 1. **Severe Speaker Imbalance** ⚠️ CRITICAL ISSUE
- Train dataset: 19:1 ratio between most and least represented speakers
- Dev dataset: 14:1 ratio between most and least represented speakers
- Will cause model bias toward overrepresented speakers
- **Recommendation**: Implement speaker-balanced sampling or data augmentation

### 2. **High Duplication Rate in Train Data**
- 163 duplicate sentences in train dataset is concerning
- May lead to overfitting on repeated content
- Consider deduplication or balanced sampling

### 3. **Significant Cross-Dataset Overlap**
- 68 overlapping sentences between train and dev
- Could lead to optimistic evaluation results
- Consider creating truly independent dev/test sets

### 4. **Duration Characteristics**
- Both datasets have similar duration distributions
- Most utterances are 2-6 seconds long
- Train dataset has some longer utterances (up to 13s)

### 5. **Data Quality Issues**
- High duplication suggests potential data collection methodology issues
- Cross-dataset overlap indicates need for better data splitting strategy
- Speaker imbalance suggests non-random data collection or selection

## Recommendations

### **High Priority:**
1. **Address speaker imbalance** - Implement speaker-balanced sampling or collect more data from underrepresented speakers
2. **Create independent dev/test sets** with no sentence overlap
3. **Investigate data collection process** to understand imbalance and duplication patterns

### **Medium Priority:**
4. **Deduplicate training data** to prevent overfitting
5. **Consider speaker-specific evaluation** given the severe imbalance
6. **Analyze phonetic content** of duplicate sentences to understand if they serve a specific purpose

### **Low Priority:**
7. **Standardize duration ranges** if needed for model training
8. **Document speaker characteristics** for better understanding of bias

## Impact on Model Training

The severe speaker imbalance will likely cause:
- **Model bias** toward overrepresented speakers (F02, F03, M01, M03)
- **Poor generalization** to underrepresented speakers
- **Inflated performance metrics** due to cross-dataset overlap
- **Unreliable evaluation** of speaker-independent performance

**Immediate action required** to address these data quality issues before model training. 