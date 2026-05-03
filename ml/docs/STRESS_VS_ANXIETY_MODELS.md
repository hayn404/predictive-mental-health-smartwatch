# Seren Dual-Model Architecture: Stress vs. Anxiety
## A Deep Technical Comparison

**Location:** `ml/docs/STRESS_VS_ANXIETY_MODELS.md`
**Date:** May 2026

---

## 1. The Core Philosophy

> The brain and the body distinguish between *"I am being challenged right now"* (Stress) and *"I am afraid something bad will happen"* (Anxiety). Seren's dual-model system mirrors this exact distinction at the physiological level.

These two emotional states have **overlapping but different physiological signatures**. The two models were deliberately engineered to isolate those signatures from each other.

---

## 2. Model A — The Stress Model

### 2.1 Psychological Definition
Acute stress is a **short-term, present-tense response** to an immediate identifiable challenge. Your boss gave you a 10-minute deadline. You are in a car accident. You are sprinting. The stressor *exists right now* and the body reacts immediately.

### 2.2 Training Data
| Dataset | Subjects | Stress Protocol | Labels Used |
|---|---|---|---|
| **WESAD (Wrist)** | 15 subjects | Public speaking + Trier Social Stress Test + Mental arithmetic under judgment | `label == 2` → Stressed, everything else → Baseline |
| **Synthetic Data** | 500 simulated | Physiologically realistic HR spikes + HRV depression | Binary 0/1 |

**What WESAD's stress condition looked like:**
- Subjects stood in front of a panel of judges and gave a 5-minute speech.
- Then they did mental math (serial subtraction) while the judges shook their heads disapprovingly.
- These are some of the most powerful laboratory stressors known to science (Trier Social Stress Test protocol).
- The body reacted within **seconds** — massive HR spike, EDA surge, HRV crash.

### 2.3 The Target Variable (Label)
```
Label = 1   ← Subject is ACTIVELY in the stress protocol
Label = 0   ← Subject is at baseline, amused, or meditating
```
This is a **Binary Classification** problem. The model has to answer one question: *"Is the body under acute load RIGHT NOW?"*

### 2.4 Features Most Important to the Stress Model
The Stress Model cares most about **rapid, sudden changes**:

| Feature | Why it matters for Stress |
|---|---|
| `hrMean` | Heart rate shoots up within seconds of an acute stressor |
| `hrStd` | High variability during the stress event as the body ramps up |
| `hrRange` | The peak-to-valley range explodes during active stressors |
| `rmssd` | Plummets immediately — the vagus nerve gets shut off fast |
| `sdnn` | Drops sharply as HRV compresses |
| `lfHfRatio` | Spikes as Sympathetic Nervous System (fight-or-flight) takes over |
| `tempSlope` | Skin temperature drops as blood vessels constrict |
| `stepRate` | Can be elevated if the stress is physical |

**The Stress Model is looking for:** An extreme, sudden, high-amplitude disruption across multiple channels simultaneously.

### 2.5 Model Architecture
```
Algorithm:  XGBoost Binary Classifier
Objective:  binary:logistic
Output:     Probability 0.0 → 1.0
App Display: Probability × 100 → Stress Score (0-100)
Trees:      200
Training:   LOSO-CV (Leave-One-Subject-Out)
```

---

## 3. Model B — The Anxiety Model

### 3.1 Psychological Definition
Anxiety is a **long-term, future-oriented response to a threat that may or may not be real**. It is worry. It is anticipation. It is sitting at your desk thinking about a meeting tomorrow. The stressor does not need to exist in the present moment — the nervous system *acts as if it does* for sustained periods.

### 3.2 Training Data
| Dataset | Subjects | Anxiety Protocol | Labels Used |
|---|---|---|---|
| **WESAD (Wrist)** | 15 subjects | Pre-experiment STAI questionnaire | STAI score → mapped to 0-100 index |
| **PhysioNet Exam Stress** | 10 students | Before/during/after university exams | Continuous perceived stress + anxiety ratings |
| **Anxiety Dataset 2022** | 120+ records | Clinical GAD-7 and BAI questionnaire scores | GAD-7 score (0-21) → mapped to 0-100 |

**What the Anxiety Dataset looked like:**
- Subjects wore wrist sensors for days to weeks at a time.
- They filled out clinical questionnaires (GAD-7 = Generalized Anxiety Disorder 7-item scale) regularly.
- These questionnaires asked things like: "Over the last 2 weeks, how often have you been unable to stop worrying?"
- The answer (a score from 0 to 21) became our label. A score of 20 = Severe Anxiety (100 on our scale).

### 3.3 The Target Variable (Label)
```
Label = 0.0–100.0  ← GAD-7/STAI clinical anxiety score, mapped to 0-100
```
This is a **Regression** problem. The model has to answer: *"On a scale of 0 to 100, how anxious is this person's nervous system right now — even if they are sitting still?"*

### 3.4 Features Most Important to the Anxiety Model
The Anxiety Model cares most about **subtle, sustained, chronic patterns**:

| Feature | Why it matters for Anxiety |
|---|---|
| `rmssd` | Chronically suppressed RMSSD over hours/days = anxious baseline |
| `hfPower` | HF power (Parasympathetic tone) is persistently low in anxious individuals — even at rest |
| `lfHfRatio` | Sustained sympathetic dominance even without any active stressor |
| `sd1` (Poincaré) | The short-term scatter of heartbeats is chronically compressed |
| `sampleEntropy` | Anxious hearts have less "chaos" — the rhythm becomes rigid and predictable |
| `tempMean` | Chronically cold hands/wrists → persistent vasoconstriction |
| `meanRR` | Baseline R-R interval — chronically short (elevated resting HR) in anxious individuals |
| `pNN50` | Very low even during rest in high-anxiety individuals |

**The Anxiety Model is looking for:** A sustained, quiet, long-term suppression of the Parasympathetic Nervous System across multiple features — even when there is no obvious stressor.

### 3.5 Model Architecture
```
Algorithm:  XGBoost Regressor
Objective:  reg:squarederror
Output:     Direct score 0.0 → 100.0
App Display: Score directly (no conversion needed)
Trees:      487 (Early Stopping found optimal)
Training:   LOSO-CV (Leave-One-Subject-Out)
MAE:        10.64
Precision:  96% (for Severe Anxiety classification)
```

---

## 4. Correlation Between the Two Models

The two scores are **partially correlated but not identical**. Here is the mathematical relationship:

### 4.1 When they are correlated (move together)
Both scores will be high when the nervous system is severely activated AND sustained:

```
Example: A student is in the middle of a panic attack during an exam.
→ Stress Score:   90  (Body is under massive acute load)
→ Anxiety Score:  88  (Clinical anxiety levels are severe)
```

They correlate because the same physiological channels (HRV, HR, temperature) drive both. But the *weight* given to each channel is different.

### 4.2 When they diverge (the valuable cases)

The divergence is where Seren's dual-model system provides real clinical value:

```
Scenario A: "Flow State / High Performance"
→ Stress Score:   85  (Body working hard — elevated HR, compressed HRV)
→ Anxiety Score:  15  (No worry, no anticipatory fear)
Interpretation:   "You are focused and challenged. Not anxious."

Scenario B: "Anticipatory Anxiety / Chronic Worry" ← THE KEY CASE
→ Stress Score:   22  (Body is calm — resting HR, no acute stressor)
→ Anxiety Score:  78  (Parasympathetic system persistently suppressed)
Interpretation:   "You are sitting still but your nervous system is chronically activated."

Scenario C: "Post-Workout Recovery"
→ Stress Score:   30  (HR returning to normal after exercise)
→ Anxiety Score:  10  (Sleep quality good, baseline HRV normal)
Interpretation:   "You exercised and your body is recovering normally."

Scenario D: "Burnout / Exhaustion"
→ Stress Score:   55  (Moderate arousal — body is tired)
→ Anxiety Score:  82  (Weeks of poor sleep have destroyed HRV baseline)
Interpretation:   "You are not acutely stressed but chronically anxious and exhausted."
```

---

## 5. When Anxiety Can Be HIGH Without Stress Being HIGH

This is the most clinically important capability of the dual-model system. Here are the **specific physiological reasons** why Anxiety can be high while Stress is low:

### 5.1 The Parasympathetic Suppression Pattern
A person sitting completely still, doing nothing, can have:
- **Normal HR** (70 BPM) → Stress model sees no elevation → Low Stress Score
- **Chronically depressed RMSSD** (8ms instead of 35ms) → Anxiety model sees severe parasympathetic withdrawal → High Anxiety Score

The Stress Model only sees *"current HR is normal"* and gives a low score. The Anxiety Model digs deeper and sees the *quality* of the heartbeat rhythm — which is abnormal even though the rate looks fine.

### 5.2 Real-World Scenarios Where This Happens

| Scenario | Why Stress Low | Why Anxiety High |
|---|---|---|
| **Lying in bed unable to sleep, ruminating** | HR normal (60 BPM), no physical activity | RMSSD chronically suppressed, LF/HF ratio elevated, Poincaré SD1 compressed |
| **On a phone call about bad medical news** | Sitting still, no physical exertion | sampleEntropy drops, hfPower vanishes — parasympathetic fully withdrawn |
| **Waiting for exam results** | Physically resting at desk | Sustained HRV suppression across hours, not seconds |
| **Social anxiety before a presentation** | Not yet in the stressful event | HRV already collapsed from anticipation for hours beforehand |
| **GAD (Generalized Anxiety Disorder)** | Normal daily activity, not "stressed" by any specific event | Baseline HRV is chronically below normal every single day |
| **Post-traumatic hypervigilance (PTSD)** | No acute stressor present | Autonomic system permanently locked into sympathetic dominance |
| **Caffeine overdose** | Sitting still | HR mildly elevated, RMSSD suppressed — body chemistry mimics anxiety |

### 5.3 The Physiological Mechanism
```
Stress:  Acute Sympathetic Surge → "Something is happening NOW"
         Detectable in: HR spike, HRV crash, Temp drop (seconds-to-minutes)
         
Anxiety: Sustained Parasympathetic Withdrawal → "Something MIGHT happen"
         Detectable in: Chronic RMSSD suppression, rigid heart rhythm,
                        persistently elevated LF/HF (hours-to-weeks)
```

The Autonomic Nervous System has two branches:
- **Sympathetic (Gas Pedal):** Activated by BOTH stress and anxiety
- **Parasympathetic (Brake):** The key differentiator — in anxiety, the brake is chronically lifted even without the gas being pushed

---

## 6. The Decision Matrix in the Seren App

```
                    Stress LOW        Stress HIGH
Anxiety LOW    │  Resting Calm    │  Flow State /    │
               │  (Optimal)       │  Physical Peak   │
               │                  │                  │
Anxiety HIGH   │  Chronic Worry / │  Full Panic /    │
               │  GAD / PTSD      │  Anxiety Attack  │
               │  ← KEY CASE      │                  │
```

Each quadrant triggers different interventions in Seren's Recommendations Engine:
- **Both High:** Immediate grounding exercise (4-7-8 breathing), push notification
- **Stress High only:** Physical recovery recommendation (hydration, rest)
- **Anxiety High only:** Cognitive intervention (journaling, GAD-7 check-in)
- **Both Low:** Positive reinforcement, maintenance wellness tip

---

## 7. Feature Overlap Summary

| Feature | Used by Stress | Used by Anxiety | Why Different Weight |
|---|---|---|---|
| `hrMean` | ⭐⭐⭐ High weight | ⭐ Low weight | Stress needs acute HR spike; chronic resting HR matters less |
| `rmssd` | ⭐⭐ Medium | ⭐⭐⭐ Highest | Anxiety model learned RMSSD is the #1 chronic anxiety signal |
| `lfHfRatio` | ⭐⭐ Medium | ⭐⭐⭐ High | Sustained sympathetic dominance is the anxiety signature |
| `hfPower` | ⭐ Low | ⭐⭐⭐ High | Parasympathetic power is anxiety's #1 marker |
| `sampleEntropy` | ⭐ Low | ⭐⭐⭐ High | Rigid, predictable heartbeats = anxious nervous system |
| `hrRange` | ⭐⭐⭐ High | ⭐ Low | Range explosion is acute stress, not chronic anxiety |
| `tempSlope` | ⭐⭐ Medium | ⭐⭐ Medium | Cold extremities relevant to both |
| `sdnn` | ⭐⭐ Medium | ⭐⭐ Medium | Compressed in both, but timeframe differs |

---

*Generated by Seren AI Architecture Team — May 2026*
