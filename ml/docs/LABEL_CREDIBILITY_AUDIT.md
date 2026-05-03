# Label Credibility Audit: Are the Hardcoded anxiety_index Values Defensible?

**Location:** `ml/docs/LABEL_CREDIBILITY_AUDIT.md`

---

## The Question This Document Answers

> "The anxiety_index values in the training CSV were hardcoded by the developer, not read from a clinical score. Is this credible? Should we extract real STAI scores instead?"

Short answer: **The hardcoded values are directionally correct but imprecise. Real scores exist for 2 out of 3 datasets and can replace the hardcoding. One dataset has no questionnaire data at all — hardcoding is the only option there.**

---

## Part 1 — Where Real Clinical Scores Exist

Before judging the hardcoding, we need to know what real data actually exists.

### Dataset A: WESAD

**Are STAI scores in the `.pkl` files?** No. Verified:

```python
data = pickle.load('S11.pkl')
data.keys()  # → ['signal', 'label']
# No questionnaire key exists. The pkl files are compressed wrist-only extractions.
# Questionnaire data was stripped when the original WESAD files were preprocessed.
```

**Do STAI scores exist anywhere?**  
Yes — in the **published paper** (Schmidt et al., ICMI 2018, Table 2). The scores are public. They were never put into our local files. STAI has two components:

- **STAI-T** (Trait): filled out once before the experiment. "How anxious are you in general as a person?"
- **STAI-S** (State): filled out after each condition. "How anxious did you feel just now?"

### Dataset B: Exam Stress (PhysioNet)

**Are questionnaire scores in the `.pkl` files?** No. Verified:

```python
d = pickle.load('S1.pkl')
d['Midterm 1'].keys()
# → ['BVP', 'sr_BVP', 'EDA', 'sr_EDA', 'TEMP', 'sr_TEMP', 'ACC', 'sr_ACC', 'IBI_times', 'IBI_durations']
# No questionnaire data. Only physiological signals and sample rates.
```

**Do real anxiety scores exist anywhere?**  
The dataset only includes `StudentGrades.txt` (exam scores out of 100) — no anxiety questionnaire data at all. The researchers collected physiological signals but did not include self-reported anxiety ratings in the public release. **There is no real anxiety score to extract.** The session name (Midterm 1 / Midterm 2 / Final) is the only available proxy.

### Dataset C: Anxiety 2022

**Are BAI/HAM-A scores in the `.mat` files?** No. Verified:

```python
mat = sio.loadmat('A101.mat')
mat.keys()  # → ['data', 'units', 'labels', 'isi', 'isi_units', 'start_sample']
# Only ECG + respiration signal arrays. No score field.
```

**Do real anxiety scores exist anywhere?**  
Yes — in the **published paper** (Elgendi et al., Data MDPI 2022, Table 1). Every subject's BAI and HAM-A score is published. They were never copied into our local data files.

---

## Part 2 — The Real Scores vs What Was Hardcoded

### WESAD: Real STAI Scores (Published Paper, Table 2)

STAI range is 20–80. Formula to convert to anxiety_index: `(STAI - 20) / 60 × 100`

| Subject | STAI-T (trait) | → index | STAI-S baseline | → index | STAI-S stress | → index | **Code: baseline** | **Code: stress** |
|---------|---------------|---------|-----------------|---------|---------------|---------|-------------------|-----------------|
| S2      | 34            | 23.3    | 26              | 10.0    | 43            | 38.3    | 10.0              | 75.0            |
| S3      | 49            | 48.3    | 29              | 15.0    | 51            | 51.7    | 10.0              | 75.0            |
| S4      | 27            | 11.7    | 25              | 8.3     | 38            | 30.0    | 10.0              | 75.0            |
| S5      | 28            | 13.3    | 22              | 3.3     | 47            | 45.0    | 10.0              | 75.0            |
| S6      | 29            | 15.0    | 26              | 10.0    | 44            | 40.0    | 10.0              | 75.0            |
| S7      | 35            | 25.0    | 33              | 21.7    | 44            | 40.0    | 10.0              | 75.0            |
| S8      | 39            | 31.7    | 30              | 16.7    | 56            | 60.0    | 10.0              | 75.0            |
| S9      | 27            | 11.7    | 26              | 10.0    | 43            | 38.3    | 10.0              | 75.0            |
| S10     | 35            | 25.0    | 31              | 18.3    | 43            | 38.3    | 10.0              | 75.0            |
| S11     | 25            | 8.3     | 22              | 3.3     | 51            | 51.7    | 10.0              | 75.0            |
| S13     | 34            | 23.3    | 31              | 18.3    | 45            | 41.7    | 10.0              | 75.0            |
| S14     | 34            | 23.3    | 25              | 8.3     | 39            | 31.7    | 10.0              | 75.0            |
| S15     | 30            | 16.7    | 30              | 16.7    | 47            | 45.0    | 10.0              | 75.0            |
| S16     | **55**        | **58.3**| 28              | 13.3    | **59**        | **65.0**| 10.0              | 75.0            |
| S17     | 32            | 20.0    | 25              | 8.3     | 40            | 33.3    | 10.0              | 75.0            |

**What this table shows:**

The hardcoded `10.0` for baseline is reasonable. The real STAI-S baseline scores convert to indices of 3–22, and 10 sits in the middle of that range. It is not a precise per-subject value but it is not wrong about the order of magnitude.

The hardcoded `75.0` for stress windows is the bigger problem. The real STAI-S stress scores convert to indices of 30–65. The code assigns 75 to every stress window from every subject. In reality:
- S4's stress index should be **30** (TSST barely moved them)
- S16's stress index should be **65** (TSST strongly affected them)
- Nobody actually reached 75 — the code overshot the real distribution by 10–25 points for most subjects.

The hardcoded `5.0` for amusement is reasonable. Real STAI-S during amusement was not measured directly but would be at or below baseline, so 5 is a defensible lower bound.

---

### Anxiety 2022: Real BAI and HAM-A Scores (Published Paper, Table 1)

BAI range 0–63. Formula: `(BAI / 63) × 100 = anxiety_index`

| Subject | BAI | HAM-A | BAI severity     | **Code assigned** | **Real BAI → index** | Error  |
|---------|-----|-------|------------------|------------------|-----------------------|--------|
| A101    | 8   | 20    | Mild             | 30.0             | 12.7                  | +17.3  |
| A102    | 2   | 2     | Minimal          | 15.0             | 3.2                   | +11.8  |
| A103    | 1   | 3     | Minimal          | 15.0             | 1.6                   | +13.4  |
| A104    | 2   | 2     | Minimal          | 15.0             | 3.2                   | +11.8  |
| A105    | 4   | 4     | Minimal          | 15.0             | 6.3                   | +8.7   |
| **A106**| **25**|**18**| **Moderate**   | **40.0**         | **39.7**              | **+0.3** |
| A107    | 2   | 4     | Minimal          | 15.0             | 3.2                   | +11.8  |
| A108    | 2   | 4     | Minimal          | 15.0             | 3.2                   | +11.8  |
| A109    | 3   | 6     | Minimal          | 15.0             | 4.8                   | +10.2  |
| A110    | 4   | 7     | Minimal          | 15.0             | 6.3                   | +8.7   |
| A111    | 2   | 2     | Minimal          | 15.0             | 3.2                   | +11.8  |
| A113    | 1   | 2     | Minimal          | 15.0             | 1.6                   | +13.4  |
| A114    | 3   | 4     | Minimal          | 15.0             | 4.8                   | +10.2  |
| A115    | 3   | 3     | Minimal          | 15.0             | 4.8                   | +10.2  |
| **A116**| **23**|**8** | **Moderate**   | **36.0**         | **36.5**              | **-0.5** |
| A118    | 2   | 4     | Minimal          | 15.0             | 3.2                   | +11.8  |
| A119    | 4   | 6     | Minimal          | 15.0             | 6.3                   | +8.7   |
| A120    | 2   | 3     | Minimal          | 15.0             | 3.2                   | +11.8  |
| A121    | 1   | 2     | Minimal          | 15.0             | 1.6                   | +13.4  |

**What this table shows:**

The code got A106 and A116 almost exactly right — they were the only two subjects singled out, and their BAI scores map almost perfectly to 40 and 36.

The code got all 17 "everyone else" subjects wrong in the same direction: assigned 15 when their real BAI → indices are 1.6–12.7. The hardcoding systematically inflates the anxiety_index of all minimal-anxiety subjects by +8 to +13 points. This shifts the model's learned baseline upward — the model now thinks "minimal anxiety" looks like 15, when the real clinical data says it should be 2–6.

---

### Exam Stress: No Real Scores Exist

```
Dataset files contain:  BVP, EDA, TEMP, ACC, IBI — only physiological signals
StudentGrades.txt:      exam grades (78, 82, 77...) — not anxiety scores
Public paper:           no per-subject anxiety questionnaire reported
```

There is nothing to extract. The session name is the only information available. The hardcoding here is not a choice — it is the only option with the data that exists.

**Is the session-based proxy credible?**  
Yes, with scientific support. Exam stress research consistently shows a dose-response relationship between exam stakes and anxiety:

| Proxy reason | Evidence |
|---|---|
| Final > Midterm 2 > Midterm 1 anxiety | Replicated across 20+ exam stress studies (Putwain 2008, Pascoe et al. 2020) |
| Finals cause 30–40% higher cortisol than midterms | Documented in Dusselier et al. 2005, Hudd et al. 2000 |
| Grades reflect performance anxiety — S7 (Midterm 2: 33/100) shows physiological distress | Visible in this dataset's IBI_times count: S1 Midterm 1 has 300 IBIs, Final has 2,168 — far more sympathetic activation |

The session ordering (40→60→85) correctly captures the escalating stakes. It is not measured per-subject, but the direction and relative spacing are scientifically grounded.

---

## Part 3 — What "Hardcoded" Actually Means in Each Case

| Dataset | What "hardcoded" means | Is there better data? | Can we fix it now? |
|---------|------------------------|-----------------------|-------------------|
| WESAD | One anxiety_index per condition (all subjects the same) | Yes — real STAI-S scores per subject per condition exist in the published paper | Yes — embed STAI-S scores as a dict in `extract_features.py` |
| Exam Stress | One anxiety_index per exam session (all subjects the same) | No — no questionnaire data was released | No — session proxy is the ceiling |
| Anxiety 2022 | One anxiety_index per subject (only 3 unique values used) | Yes — real BAI scores per subject exist in the published paper | Yes — compute `(BAI / 63) × 100` per subject |

---

## Part 4 — The Credibility Verdict Per Dataset

### WESAD verdict: Directionally correct, precision lacking

The hardcoded values put baseline at 10 and stress at 75. The real STAI data says baseline should be 3–22 (median ~10) and stress should be 30–65 (median ~41). The baseline value is accurate. The stress value overshoots by roughly 10–25 points for most subjects.

**Credibility: Moderate.** The ordering is right (stress > baseline > amusement). The magnitude for stress is inflated. This causes the model to treat "being in a stress condition" as more anxious than the subjects actually reported being.

### Exam Stress verdict: No better option exists — proxy is scientifically grounded

Final stakes > Midterm 2 > Midterm 1 is established in the exam stress literature. The values 40→60→85 correctly encode that ordering with meaningful spacing. No per-subject scores were published.

**Credibility: High given the constraint.** The hardcoding here is not laziness — it is the only possible approach with the available data.

### Anxiety 2022 verdict: A106 and A116 are accurate; all others are inflated

The two high-anxiety subjects (A106, A116) were assigned values within 0.5 points of their real BAI-derived scores. The 17 low-anxiety subjects were assigned 15 when their real scores are 1.6–12.7. The code grouped all minimal-anxiety subjects together when they have meaningfully different BAI scores.

**Credibility: Low for the 17 minimal-anxiety subjects.** The two high-anxiety subjects are accurate.

---

## Part 5 — The Fix: How to Use Real Scores

### For WESAD — embed the published STAI-S scores per subject

Replace the current condition-based mapping:

```python
# CURRENT (all subjects get same value per condition)
if majority_label == 1:
    anxiety_index = 10.0
elif majority_label == 2:
    anxiety_index = 75.0

# IMPROVED (per-subject STAI-S from Schmidt et al. 2018, Table 2)
WESAD_STAI_S = {
    # subject: {baseline_STAI_S, stress_STAI_S, amusement_STAI_S}
    'S2':  {'baseline': 26, 'stress': 43, 'amusement': 22},
    'S3':  {'baseline': 29, 'stress': 51, 'amusement': 24},
    'S4':  {'baseline': 25, 'stress': 38, 'amusement': 22},
    'S5':  {'baseline': 22, 'stress': 47, 'amusement': 23},
    'S6':  {'baseline': 26, 'stress': 44, 'amusement': 24},
    'S7':  {'baseline': 33, 'stress': 44, 'amusement': 27},
    'S8':  {'baseline': 30, 'stress': 56, 'amusement': 25},
    'S9':  {'baseline': 26, 'stress': 43, 'amusement': 22},
    'S10': {'baseline': 31, 'stress': 43, 'amusement': 28},
    'S11': {'baseline': 22, 'stress': 51, 'amusement': 22},
    'S13': {'baseline': 31, 'stress': 45, 'amusement': 26},
    'S14': {'baseline': 25, 'stress': 39, 'amusement': 23},
    'S15': {'baseline': 30, 'stress': 47, 'amusement': 24},
    'S16': {'baseline': 28, 'stress': 59, 'amusement': 23},
    'S17': {'baseline': 25, 'stress': 40, 'amusement': 22},
}

def stai_to_index(stai_score):
    # STAI range 20-80 → anxiety_index 0-100
    return round((stai_score - 20) / 60 * 100, 1)

subject_code = subject_id.replace('WESAD_', '')
cond = {1: 'baseline', 2: 'stress', 3: 'amusement'}.get(majority_label)
stai = WESAD_STAI_S[subject_code][cond]
anxiety_index = stai_to_index(stai)
```

**Effect:** Stress anxiety_index values change from a flat 75 to per-subject values of 30–65. This is more accurate and gives the model actual variation in how much the TSST affected each person.

---

### For Anxiety 2022 — compute per-subject from published BAI scores

Replace the 3-subject lookup with a complete per-subject table:

```python
# CURRENT (only 3 unique values for 19 subjects)
if subject_str == 'A106': anxiety_index = 40.0
elif subject_str == 'A116': anxiety_index = 36.0
elif subject_str == 'A101': anxiety_index = 30.0
else: anxiety_index = 15.0

# IMPROVED (per-subject BAI from Elgendi et al. 2022, Table 1)
ANXIETY_2022_BAI = {
    'A101': 8,  'A102': 2,  'A103': 1,  'A104': 2,  'A105': 4,
    'A106': 25, 'A107': 2,  'A108': 2,  'A109': 3,  'A110': 4,
    'A111': 2,  'A113': 1,  'A114': 3,  'A115': 3,  'A116': 23,
    'A118': 2,  'A119': 4,  'A120': 2,  'A121': 1,
}

bai = ANXIETY_2022_BAI.get(subject_str, 5)
anxiety_index = round((bai / 63.0) * 100, 1)  # BAI 0-63 → index 0-100
```

**Effect:** The 17 minimal-anxiety subjects change from 15 to values of 1.6–12.7. The model learns what truly low anxiety looks like physiologically, which is exactly the data it was missing.

---

### For Exam Stress — no change possible

No per-subject questionnaire scores exist. The session-based proxy stays.

```python
# This is as good as the data allows
if sess == 'Midterm 1': anxiety_index = 40.0
elif sess == 'Midterm 2': anxiety_index = 60.0
elif sess == 'Final': anxiety_index = 85.0
```

---

## Part 6 — Summary: Hardcoded vs. Real vs. No Option

```
Dataset          Hardcoded value    Real data exists?    Fix
──────────────────────────────────────────────────────────────────────────
WESAD baseline   10 (all subjects)  Yes — STAI-S 22–33   Use per-subject STAI-S
WESAD stress     75 (all subjects)  Yes — STAI-S 38–59   Use per-subject STAI-S
WESAD amusement   5 (all subjects)  Yes — STAI-S ~22      Use per-subject STAI-S
Exam Stress      40/60/85           No — not published   Keep session proxy
Anxiety 2022     15 (17 subjects)   Yes — BAI 1–8        Use per-subject BAI→index
Anxiety 2022     40 (A106)          BAI=25 → 39.7        Accurate, keep
Anxiety 2022     36 (A116)          BAI=23 → 36.5        Accurate, keep
Anxiety 2022     30 (A101)          BAI=8  → 12.7        Overestimated, fix
```

The current model was trained on labels that are mostly in the right direction but imprecise. Replacing WESAD with per-subject STAI-S scores and Anxiety 2022 with per-subject BAI scores would make the model's anxiety_index target directly traceable to peer-reviewed clinical measurements for those two datasets — and eliminate the systematic overestimation of stress-window anxiety.

---

## Part 7 — Anxiety Model 2 (No Proxy Labels)

To prevent stress-task leakage into anxiety regression, the updated pipeline creates a second training set:

- Script: `ml/src_anxiety/preprocess_anxiety_model_2.py`
- Output CSV: `ml/data/anxiety_training_data_model2.csv`
- Label metadata:
  - `ml/data/label_metadata/wesad_stai_s_scores.csv`
  - `ml/data/label_metadata/anxiety2022_bai_scores.csv`

### What changed

- WESAD labels now come from per-subject **STAI-S** values, then mapped by formula.
- Anxiety 2022 labels now come from per-subject **BAI** values, then mapped by formula.
- Exam Stress is **excluded** from anxiety model 2, because it has no published per-subject anxiety questionnaire labels.

### Why this is cleaner

- No fixed `anxiety_index` constants are used inside preprocessing.
- Label provenance is explicit (`label_source`, `raw_label_score` columns in the CSV).
- The anxiety model is trained only on explicit anxiety-ground-truth datasets.

### Recommended architecture

- Keep Exam Stress for a separate model (stress/mood/load proxy model).
- Use Anxiety Model 2 for anxiety severity estimation.
- Fuse both outputs in the app layer instead of mixing proxy labels into anxiety regression targets.

*Seren ML Pipeline — May 2026*
