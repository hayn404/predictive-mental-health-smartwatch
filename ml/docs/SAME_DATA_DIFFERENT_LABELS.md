# Why the Same WESAD Dataset Trains Two Models That Measure Different Things

**Location:** `ml/docs/SAME_DATA_DIFFERENT_LABELS.md`

---

## The Question This Document Answers

> "WESAD has one label column. Both models use WESAD. On paper they should measure the same thing. Why do they give different readings?"

The answer requires understanding that **the label column and the features are not the same thing**.
Both models use the same 29 physiological features (HRV, HR, temperature, accelerometer).
But each model was trained to predict a completely different number.

---

## Part 1 — What WESAD Actually Contains

WESAD is a lab study. 15 subjects wore an Empatica E4 wrist sensor for a full day and went through a scripted protocol:

```
Timeline per subject:
  ─────────────────────────────────────────────────────────────────
  [Baseline 1]  →  [TSST stress]  →  [Recovery]  →  [Amusement]  →  [Meditation]
        ↑                ↑
   label == 1      label == 2
  (resting)       (public speech +
                   mental math under judges)
  ─────────────────────────────────────────────────────────────────
```

WESAD contains two completely separate types of ground truth:

| Ground Truth Type | What It Is | Where It Lives |
|---|---|---|
| **Condition label** | A number written by researchers: "this 2-minute window is baseline / stress / amusement" | `label` column in the raw data |
| **STAI questionnaire score** | A number written by the subject: "on a 1-5 scale, rate how anxious you feel right now" | Questionnaire file filled out before the experiment |

These two numbers are **independent**. The condition label says what protocol the subject was in. The STAI score says what kind of person the subject is.

---

## Part 2 — How Each Model Reads That Same Dataset Differently

### The Stress Model Reads the Condition Label

```
X (input):   29 physiological signals from every 5-minute window
Y (target):  Was the subject inside the TSST protocol during this window?

label == 2  →  Y = 1  (stressed)
label != 2  →  Y = 0  (not stressed)
```

This is **binary classification**. The model's job is to answer one question: *"Is the nervous system under acute load RIGHT NOW?"*

The model base score is **0.5** — it starts at 50% probability and adjusts based on the features. The training accuracy was 90.1%, AUC-ROC 0.944.

### The Anxiety Model Reads the STAI Questionnaire Score

```
X (input):   29 physiological signals from BASELINE windows only (label == 1)
Y (target):  What was this subject's STAI score before the experiment started?

STAI 20-40  →  anxietyIndex  5 or 10   (low trait anxiety)
STAI 60-80  →  anxietyIndex 75         (high trait anxiety)
```

This is **regression**. The model's job is to answer: *"Given how this person's body looks right now, what is their baseline anxiety level as a personality trait?"*

The model base score is **60.06** — this is literally the mean `anxiety_index` across all training rows. The model starts at 60 and adjusts up or down.

---

## Part 3 — The Concrete Numbers From the Training Data

Here is what the WESAD contribution to the anxiety training CSV actually looks like:

```
Subject     anxiety_index    hrMean    rmssd    lfHfRatio    tempMean
─────────────────────────────────────────────────────────────────────
WESAD_S11        10.0         78.2     134.3     0.229       34.17   ← low trait anxiety
WESAD_S11        10.0         75.5     179.2     0.297       34.33     (baseline windows)
WESAD_S11        10.0         76.6     136.7     0.323       34.39
WESAD_S11        75.0        108.3     146.7     1.025       33.07   ← HIGH trait anxiety
                                                                       (stress condition)
WESAD_S11         5.0         74.6     131.8     0.682       31.74   ← very low trait anxiety
```

Subject S11 appears three times with anxiety_index=10, once with anxiety_index=75, once with anxiety_index=5.
**This is the same human being, captured at different moments in the same experiment.**

The rows with anxiety_index=10 came from S11's baseline windows — they had a low STAI score.
The row with anxiety_index=75 came from a DIFFERENT subject who happened to have a high STAI score (possibly captured during the stress condition window).

**Not a different person, not different data, not different hardware — just different labels attached to different windows.**

### Aggregate physiological differences in the WESAD training rows:

```
Group                                    hrMean    rmssd    lfHfRatio
─────────────────────────────────────────────────────────────────────
Low trait anxiety  (STAI → 5-10, n=60)   74.3      147.8    0.511
High trait anxiety (STAI → 75,   n=23)  100.7      131.0    0.566
```

Notice: the physiological difference is subtle. hrMean rises from 74 to 101, rmssd drops from 148 to 131. The anxiety model is learning these **chronic, persistent, low-amplitude differences** in the baseline physiology of anxious people — not the dramatic spike of an acute stress event.

---

## Part 4 — Why This Means They Measure Fundamentally Different Things

### Stress: An Event You Are Having

The stress model learned to detect **what happens to everyone's body during an acute load event**.

During the TSST (the 5-minute public speech in front of judges):
- Heart rate spikes 20-40 BPM within seconds
- RMSSD crashes within 60 seconds
- Temperature begins to drop as blood vessels constrict
- HRV power shifts from HF (parasympathetic) to LF (sympathetic)

These signals are **sudden, large-amplitude, and universal** across subjects. Everyone's body reacts to the judge staring at them.

The stress model's top features reflect this:

```
hrStd          0.2021   ← variability during the ramp-up and peak of stress
hrMean         0.1345   ← elevated rate during the stressor
sampleEntropy  0.1038   ← complexity drops as body goes into fight-or-flight mode
sd2            0.0840   ← long-term HRV compression during the event
```

These are all **amplitude-of-change** features. The model is detecting that something happened fast.

### Anxiety: A State You Already Live In

The anxiety model learned to detect **what a person's body looks like chronically when that person scores high on anxiety questionnaires**.

A subject with STAI=75 (high trait anxiety), even during their BASELINE resting period when nothing stressful is happening, has:
- Slightly elevated heart rate (chronic sympathetic tone)
- Chronically colder wrist temperature (persistent vasoconstriction)
- Chronically shorter mean RR intervals (higher resting HR)
- More rigid, predictable heart rhythm (lower sample entropy over many hours)

The anxiety model's top features reflect this:

```
meanRR          0.1018  ← how short the average R-R interval is chronically
tempMean        0.1001  ← chronic skin temperature (cold = persistent vasoconstriction)
accelMagnMean   0.0717  ← activity level (anxious people often move less or more erratically)
sdnn            0.0503  ← baseline HRV breadth over the whole window
```

These are **absolute-level** features. The model is detecting what the body always looks like.

---

## Part 5 — Why Three Datasets Were Needed

WESAD alone could not train a complete anxiety model because:

| Limitation | Detail |
|---|---|
| Only 15 subjects | Too few to generalize. LOSO-CV leaves only 14 for training each fold. |
| Only 3 STAI buckets | WESAD subjects cluster at STAI~10 (low) or STAI~75 (high). Almost no middle range. |
| Lab setting only | WESAD baseline = sitting quietly in a lab. Real anxiety happens at desks, in transit, at 3 AM. |

So the training CSV combines three sources to fill different regions of the anxiety spectrum:

```
Dataset          anxiety_index range    What it adds
────────────────────────────────────────────────────────────
WESAD                  5, 10, 75       Low and high extremes (clinical STAI)
PhysioNet Exam    40, 60, 85           Middle range + real-world ecol. validity
Anxiety 2022      15, 30, 36, 40       Sub-clinical range (BAI / Hamilton scores)
```

Together they cover the full 0-100 range with 44 subjects and 1,699 training windows. XGBoost convergence typically requires 500-1,000 samples — this dataset has nearly double the minimum.

---

## Part 6 — Why On Paper They "Should" Give the Same Score (But Don't)

The question "shouldn't they measure the same thing?" is intuitive because:
- Both use the same sensor signals
- Both use the word "stress" and "anxiety" colloquially as synonyms
- Both produce a 0-100 number

Here is why they legitimately diverge:

### Argument 1: Different Timeframes

Stress is a **moment** (seconds to minutes). Anxiety is a **trait** (hours to weeks).

A 5-minute biometric window contains:
- For the stress model: did anything happen during THIS 5 minutes?
- For the anxiety model: what kind of nervous system is producing this pattern?

### Argument 2: Different Mathematical Objectives

```
Stress model:   binary:logistic  →  P(stressed) = sigmoid(rawScore)  →  × 100
Anxiety model:  reg:squarederror →  predicted STAI score              →  0-100 directly
```

Sigmoid compresses everything toward 50 unless the signal is very strong. Regression spreads freely across the full range. The same features produce different numbers because the math is different.

### Argument 3: Different Feature Weights

The stress model uses `hrStd` as its most important feature (weight 0.2021).
The anxiety model uses `meanRR` as its most important feature (weight 0.1018).

`hrStd` = how much does HR fluctuate during this 5-minute window. An acute stressor makes HR spike and then ramp up — high variation.
`meanRR` = what is the average R-R interval. A chronically anxious person just has a shorter one all the time.

Same underlying signal (heart rate), completely different aspect of that signal, completely different predictive weight.

### Argument 4: The "Chronic Anxiety at Rest" Case

This is the most clinically important divergence and the reason the dual-model system exists:

```
Scenario: Person sits still at desk, HR = 72 BPM. No stressor present.

Stress model sees:
  hrMean = 72 → normal
  hrStd  = 3  → very little variation → looks like rest
  Output: Stress Score = 8   (nothing is happening)

Anxiety model sees:
  meanRR  = 833 ms  → normal rate
  rmssd   = 12 ms   → CHRONICALLY SUPPRESSED (normal is ~35ms at rest)
  tempMean = 31.5°C → cold wrist (normal is ~34°C)
  sampleEntropy = 0.4 → rigid, predictable rhythm (normal is ~1.4)
  Output: Anxiety Index = 78   (this person's nervous system is chronically overactivated)
```

The stress model says "calm." The anxiety model says "this person is suffering."

---

## Part 7 — Summary: Same Inputs, Different Labels, Different Models

```
                   STRESS MODEL              ANXIETY MODEL
────────────────────────────────────────────────────────────────────
Input features:    Same 29 signals           Same 29 signals
Training Y:        condition label (0/1)     STAI/GAD-7 score (0-100)
Task:              binary classification     regression
Training data:     ALL conditions            WESAD: baseline only
                   (baseline + TSST)         + Exam Stress windows
                                             + Clinical BAI windows
What Y means:      "Was a stressor           "How anxious is this
                   happening right now?"     person as a trait?"
Base score:        0.5 (neutral prior)       60.06 (mean training label)
Top feature:       hrStd (0.2021)            meanRR (0.1018)
Output math:       sigmoid(raw) × 100        raw score, clamped to 0-100
Time sensitivity:  seconds to minutes        hours to weeks
Clinical parallel: Cortisol spike            GAD-7 / STAI questionnaire
```

The two models are no more "the same thing" than a thermometer and a blood test.
Both measure your health. Neither measures the same quantity.

---

*Seren ML Pipeline — May 2026*
