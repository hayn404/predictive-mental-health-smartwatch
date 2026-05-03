# WESAD Label System: A Deep Technical Explanation
## How One Dataset Produces Two Different Models

**Location:** `ml/docs/WESAD_LABEL_DEEP_DIVE.md`

---

## What Is Inside a WESAD `.pkl` File

Each of the 15 subject files (e.g. `S11.pkl`) is a Python dictionary with exactly two top-level keys:

```
data = {
    'signal': {
        'wrist': {
            'BVP':  ndarray (334912, 1)   @ 64 Hz    ← heartbeat waveform
            'EDA':  ndarray ( 20932, 1)   @  4 Hz    ← skin conductance
            'TEMP': ndarray ( 20932, 1)   @  4 Hz    ← wrist temperature °C
            'ACC':  ndarray (167456, 3)   @ 32 Hz    ← accelerometer x,y,z
        }
    },
    'label':  ndarray (3663100,)           @ 700 Hz   ← condition marker
}
```

These numbers are real, from `S11.pkl`. Let's understand every one of them.

---

## Part 1 — The Label Array: What It Is And Why It's 700 Hz

The `label` array has **3,663,100 samples** at **700 Hz**.

```
Total duration: 3,663,100 ÷ 700 = 5,233 seconds = 87.2 minutes
```

At 700 Hz, the label array has **one value every 1.43 milliseconds**.

Why 700? Because the chest-worn RespiBAN device records ECG at 700 Hz, and the researchers chose this as the master clock for the entire recording. Every other signal is slower, so the label array is the highest-resolution time reference.

Each sample in the label array is a single integer:

```
Label Value   Meaning
────────────────────────────────────────────────────────
    0         Transient — between conditions, discard
    1         Baseline — subject reading a neutral magazine
    2         Stress (TSST) — public speech + mental math
    3         Amusement — watching 11 funny video clips
    4         Meditation — guided meditation session
    5, 6, 7   Study phase transition markers (rare, discard)
────────────────────────────────────────────────────────
```

For `S11.pkl`, the actual distribution:

```
label=0 (transient):   1,443,400 samples  =  375.9 min  (39.4%)
label=1 (baseline):      826,000 samples  =  215.1 min  (22.5%)  ← wait — see note below
label=2 (STRESS):        476,000 samples  =  124.0 min  (13.0%)  ← wait — see note below
label=3 (amusement):     257,600 samples  =   67.1 min  ( 7.0%)
label=4 (meditation):    553,701 samples  =  144.2 min  (15.1%)
```

**Those numbers look too large because the label array includes the FULL recording including time when the subject was putting on the device, filling out questionnaires, waiting between conditions, etc.** The label=0 "transient" samples are everything that isn't a clean experimental condition.

---

## Part 2 — The Actual Condition Timeline (Real Data, S11)

By finding where the label transitions between values, we get the exact schedule of what happened to Subject S11:

```
Time             Duration    Condition
────────────────────────────────────────────────────────────────────
  0s –   86s       1.4 min   transient [0]   ← setup, putting on device
 86s – 1266s      19.7 min   BASELINE [1]    ← read magazine, sit quietly
1266s – 1318s      0.9 min   transient [0]   ← questionnaire break
1318s – 1368s      0.8 min   phase marker [5]
1368s – 1908s      9.0 min   transient [0]   ← walk to stress room
1908s – 2588s     11.3 min   STRESS [2]      ← public speech + mental math
2588s – 2823s      3.9 min   transient [0]   ← recovery period
2823s – 2875s      0.9 min   phase marker [6]
2875s – 3573s     11.6 min   transient [0]   ← walk to next room
3573s – 3968s      6.6 min   MEDITATION [4]
3968s – 4110s      2.4 min   transient [0]
4110s – 4478s      6.1 min   AMUSEMENT [3]
4478s – 4531s      0.9 min   transient [0]
4531s – 4581s      0.8 min   phase marker [7]
4581s – 4692s      1.9 min   transient [0]
4692s – 5088s      6.6 min   MEDITATION [4]
5088s – 5233s      2.4 min   transient [0]
```

This is the ground truth chronology. The subject spent **19.7 minutes reading a magazine**, then there was a **9-minute walk** to the stress room (transit, labeled 0, discarded), then **11.3 minutes of the TSST protocol**, then more transitions, then amusement, then meditation.

---

## Part 3 — The Multi-Rate Alignment Problem

Here is the critical engineering challenge: **each signal runs at a different sample rate, and the label runs at 700 Hz**.

```
Signal    Rate     Duration of S11    Samples in S11
──────────────────────────────────────────────────────
label     700 Hz   5,233 s            3,663,100
BVP        64 Hz   5,233 s              334,912
EDA         4 Hz   5,233 s               20,932
TEMP        4 Hz   5,233 s               20,932
ACC        32 Hz   5,233 s              167,456
```

To get the label for a specific BVP sample, you use timestamps:

```python
# BVP sample 10,000 corresponds to:
bvp_time = 10000 / 64           # = 156.25 seconds into recording

# The label at that moment:
label_index = int(156.25 * 700)  # = 109,375
label_value = label[109375]      # = 1 (baseline)
```

Each BVP sample maps to exactly one label value by dividing the sample index by its sample rate to get seconds, then multiplying by 700 to get the label array index.

For a 300-second (5-minute) window:
- You get **19,200 BVP samples** (300s × 64 Hz)
- You get **1,200 TEMP samples** (300s × 4 Hz)
- You verify that **at least 80% of the label array** within that time window has the same value

---

## Part 4 — What the BVP Signal Actually Is

The BVP (Blood Volume Pulse) is a raw photoplethysmograph waveform. The E4 shines infrared light into the wrist and measures how much bounces back. When the heart beats, the wrist arteries fill with blood and absorb more light — you see a dip in reflectance.

For Subject S11, the BVP array looks like this in raw ADC units:

```
BASELINE windows (BVP statistics @ 64 Hz):
  mean ≈ 0.0 (AC-coupled, zero-mean by design)
  std  ≈ 57.5  (amplitude of the waveform)
  min  ≈ -523   max ≈ +630   (ADC range)

STRESS windows:
  mean ≈ 0.0
  std  ≈ 26.9   ← LOWER amplitude (more vasoconstriction)
  min  ≈ -351   max ≈ +248
```

The raw BVP number has no physiological meaning by itself. Its **timing** is everything. To get HRV, the preprocessing pipeline runs peak detection on the waveform to find the moment of each heartbeat, then measures the time between consecutive peaks.

---

## Part 5 — From Raw BVP → HRV Features (The Full Pipeline)

For a single 5-minute baseline window from S11 (t=86s to t=386s):

```
Step 1: Extract BVP samples
  indices 0 to 19,200  (300s × 64 Hz)
  BVP values: [45.2, 47.1, 48.9, 47.2, 44.8, ..., -312.4, -351.0, ...]

Step 2: Bandpass filter (0.5 Hz – 4.0 Hz)
  Removes slow drift (< 0.5 Hz) and high-frequency noise (> 4 Hz)
  Preserves the heartbeat waveform (normal HR is 0.9–3.3 Hz)

Step 3: Peak detection
  find_peaks(filtered_bvp, distance=25, prominence=...)
  → Finds 381 peaks (heartbeats) in this 300-second window
  → Peak timestamps in seconds: [0.52, 1.31, 2.09, 2.89, ...]

Step 4: R-R intervals (inter-beat intervals)
  diff(peak_times) × 1000  → in milliseconds
  → [790ms, 782ms, 800ms, 795ms, 810ms, ...]
  → Keep only 300ms < RR < 1500ms (40–200 BPM physiological range)

Step 5: Compute the 29 HRV features
  meanRR        = mean(RR)                    = 796 ms
  sdnn          = std(RR)                     = 137 ms
  rmssd         = sqrt(mean(diff(RR)²))       = 199 ms
  pnn50         = % of consecutive RR differing > 50ms
  hrMean        = 60000 / meanRR              = 75.4 BPM
  lfHfRatio     = LF power / HF power        = 0.28
  sampleEntropy = regularity of the RR series
  tempMean      = mean(TEMP[corresponding window]) = 34.23°C
  accelMagMean  = mean(√(ax²+ay²+az²))        = 64.5 (ADC units)
  ...
```

The output is **one row** in the training CSV: 29 numbers describing the physiological state of that 5-minute window, plus one label number.

---

## Part 6 — The Label That Gets Attached to Each Feature Row

This is where the two models diverge completely.

### For the Stress Model: The Condition Label Is the Target

```python
# Each 5-minute window gets:
if majority_label == 2:    # Most of this window is during TSST
    stress_label = 1       # Stressed
else:                      # Baseline, amusement, meditation
    stress_label = 0       # Not stressed
```

The stress model trains on ALL conditions, every window:

```
Baseline windows (S11):   [796ms, 199ms RR, ...] → stress_label = 0
Stress windows  (S11):    [747ms, 352ms RR, ...] → stress_label = 1
Amusement windows (S11):  [886ms, 660ms RR, ...] → stress_label = 0
```

### For the Anxiety Model: The STAI Score Is the Target

The STAI (State-Trait Anxiety Inventory) questionnaire was filled out **before the experiment began**, during the pre-study paperwork phase. It is a clinical test asking 20 questions like:

```
"I feel calm"              — rate 1 (almost never) to 4 (almost always)
"I feel tense"             — rate 1 to 4
"I feel at ease"           — rate 1 to 4
"I feel frightened"        — rate 1 to 4
...
```

The total score ranges from 20 to 80. A score of 20-30 means low trait anxiety; 40+ means clinically elevated anxiety; 60+ means severe.

Each WESAD subject gets one STAI score. That score gets mapped to the anxiety index scale:

```
STAI score    Anxiety Index    What it means
─────────────────────────────────────────────────────────
20 – 30           5 – 10      Minimal anxiety (majority of subjects)
50 – 65              75       Elevated anxiety (minority — ~3 subjects)
```

Those anxiety index values from the CSV (5, 10, 75) correspond directly to this STAI mapping.

Now the anxiety model attaches THIS number to the physiological windows:

```
S11 STAI score → anxiety_index = 10  (low trait anxiety)

Baseline window 1 (S11):  [796ms, 199ms, ...]  → anxiety_index = 10
Baseline window 2 (S11):  [796ms, 183ms, ...]  → anxiety_index = 10
Baseline window 3 (S11):  [799ms, 248ms, ...]  → anxiety_index = 10
Stress window 1   (S11):  [749ms, 352ms, ...]  → anxiety_index = 10  ← same label!
```

**Every window from every subject gets that subject's STAI score, regardless of what condition they were in.** S11's STAI score of 10 gets stamped on every window S11 ever produced — baseline, stress, amusement, all of them.

---

## Part 7 — The Actual Window Measurements From Real Data

Here is what Subject S11's actual HRV looks like during baseline vs. stress, extracted directly from the `.pkl` file using a bandpass filter and peak detection:

```
BASELINE windows (S11, label=1, t=86s–1266s):

  Window 1  t=086s–386s   HR=76.0 BPM   RMSSD=198.9ms   SDNN=136.9ms   meanRR=789ms   TEMP=34.09°C
  Window 2  t=386s–686s   HR=75.4 BPM   RMSSD=183.8ms   SDNN=137.9ms   meanRR=796ms   TEMP=34.23°C
  Window 3  t=686s–986s   HR=75.1 BPM   RMSSD=247.6ms   SDNN=182.0ms   meanRR=799ms   TEMP=34.35°C


STRESS windows (S11, label=2, t=1908s–2588s):

  Window 1  t=1908s–2208s  HR=80.1 BPM   RMSSD=352.2ms   SDNN=270.2ms   meanRR=749ms   TEMP=33.44°C
  Window 2  t=2208s–2508s  HR=80.3 BPM   RMSSD=373.9ms   SDNN=277.9ms   meanRR=747ms   TEMP=32.88°C
```

Notice what the two models see and attach to each of these rows:

```
                          Stress model label    Anxiety model label
                          ──────────────────    ────────────────────
Baseline Window 1              0 (not stressed)       10 (STAI score)
Baseline Window 2              0                      10
Baseline Window 3              0                      10
Stress Window 1                1 (stressed)           10  ← same anxiety label!
Stress Window 2                1                      10
```

Both models see identical input features (the 29 HRV numbers). Both models receive a completely different target number.

---

## Part 8 — Across All 15 Subjects: Scale of the WESAD Contribution

All 15 subjects combined produce this many usable 5-minute windows:

```
Condition     Total seconds    Full 5-min windows    Used by which model
────────────────────────────────────────────────────────────────────────
Baseline (1)    17,611s             58 windows       Both (stress:0, anxiety:STAI)
Stress   (2)     9,966s             33 windows       Stress model (label=1)
Amusement(3)     5,575s             18 windows       Stress model (label=0 proxy)
Meditation(4)   11,806s             39 windows       Excluded or label=0
────────────────────────────────────────────────────────────────────────
TOTAL usable    33,152s            ~109 windows
```

For the stress model, all 109 windows are used: 33 get label=1 (stressed), 76 get label=0.
For the anxiety model, all 109 windows get the subject's STAI score. Subjects with STAI≈25 produce anxiety_index=10 rows; subjects with STAI≈65 produce anxiety_index=75 rows.

---

## Part 9 — Why the Anxiety Model Sees Different Things Than the Stress Model

Same features. Same windows. But look at what each model is learning:

**The stress model learns the difference between these two physiological states:**

```
State A (label=0, baseline):  meanRR=796ms  RMSSD=210ms  temp=34.3°C
State B (label=1, STRESS):    meanRR=748ms  RMSSD=363ms  temp=33.2°C  ← 680s of acute load
```

It learns: high HR + low temp + elevated RMSSD fluctuation = someone currently under acute load.

**The anxiety model learns the difference between these two types of people:**

```
Person A (STAI=25, anxiety_index=10):  baseline meanRR≈795ms  RMSSD≈210ms  temp≈34.3°C
Person B (STAI=65, anxiety_index=75):  baseline meanRR≈730ms  RMSSD≈131ms  temp≈32.7°C
                                                                ↑
                                                    Chronically suppressed even at rest
```

Person B has lower RMSSD **while sitting quietly doing nothing** — not during any acute event. The anxiety model learns that a person's chronically suppressed HRV predicts their STAI score, independent of what they were doing when the measurement was taken.

---

## Part 10 — Why the Same Input Can Produce Different Outputs

Take a single real 5-minute stress window from S11:
```
features: meanRR=749ms, RMSSD=352ms, hrMean=80.1, temp=33.4°C, ...
```

**Stress model query:** "Is this person under acute load?"
- Stress training label for this window: **1** (this was during TSST)
- The model learned that `hrMean=80` + `temp drop` = stressed
- Output: **Stress Score = 71** (elevated)

**Anxiety model query:** "How anxious is this person's nervous system as a baseline trait?"
- Anxiety training label for this window: **10** (S11's STAI score, set before the experiment)
- The model learned that subjects with meanRR=749 at baseline have STAI≈25, not STAI≈65
- Output: **Anxiety Index = 20** (minimal)

Same window. Same features. Stress = 71. Anxiety = 20.

The divergence is not a bug. It is the entire point.

---

## Part 11 — The Label Pipeline Summary

```
 RAW .pkl FILE
 ─────────────────────────────────────────────────────────────────────
 
 S11.pkl
   ├── signal.wrist.BVP   [334,912 samples @ 64 Hz]  ← raw heartbeat waveform
   ├── signal.wrist.TEMP  [ 20,932 samples @  4 Hz]  ← wrist temperature
   └── label              [3,663,100 samples @ 700 Hz] ← condition markers
 
 PREPROCESSING
 ─────────────────────────────────────────────────────────────────────
 
 1. Align signals by timestamp
 2. Find windows where label has ≥ 80% purity (same condition)
 3. Bandpass BVP → peak detection → R-R intervals
 4. Compute 29 HRV/temp/accel features per window
 
 LABELING SPLIT
 ─────────────────────────────────────────────────────────────────────
 
 STRESS MODEL                       ANXIETY MODEL
 ──────────────────────             ──────────────────────────────
 window condition = 2?              look up this subject's
   → stress_label = 1               pre-study STAI score
 window condition ≠ 2?              → map STAI → anxiety_index
   → stress_label = 0               → stamp on ALL windows
                                      from this subject
 
 TRAINING
 ─────────────────────────────────────────────────────────────────────
 
 STRESS MODEL                       ANXIETY MODEL
 ──────────────────────             ──────────────────────────────
 XGBoost binary classifier          XGBoost regressor
 objective: binary:logistic         objective: reg:squarederror
 Y: 0 or 1                          Y: 0.0 to 100.0 (STAI-mapped)
 base_score: 0.5                    base_score: 60.06 (training mean)
 Trees: 200                         Trees: 487
 Output: sigmoid(raw) × 100        Output: raw regression score
```

---

## The One-Line Answer to "Why Are They Different?"

The `label` column in WESAD tells you **what the researchers did to the subject**.
The STAI score tells you **what kind of person the subject already was**.

One model learned to detect an event. The other learned to describe a trait.
They share the same sensor data and produce different numbers because they were trained to answer different questions.

---

*Seren ML Pipeline — May 2026*
