# Predictive Mental Health Monitoring via AI-Powered Smartwatches

## Overview

This project aims to develop an AI-powered system that continuously monitors mental health using smartwatch data and generative AI technologies. The system tracks heart rate variability (HRV), sleep, physical activity, light exposure, and stress, combining them with voice-based mood assessments in both English and Arabic. By learning each user's personal patterns, the system predicts emerging mental health issues (e.g., anxiety, depression, burnout) and provides timely alerts or recommendations for early intervention.

## Key Features

- Real-time HRV, sleep, and activity monitoring
- Machine learning predictions of depression, anxiety, and stress episodes
- Conversational AI (speech/text)—mood check-ins, symptom extraction, daily journal (Arabic and English support)
- Burnout risk, exam anxiety, and crisis detection (including suicide risk/protective alerts)
- Simple dashboard and data visualization
- Privacy-first design: encrypted, user-controllable, no personal data sharing by default

## Repository Structure
predictive-mental-health-smartwatch/
├── README.md
├── LICENSE
├── requirements.txt
├── docs/                   # Proposal, reports, user docs
├── data/                   # Example + processed datasets (do not share personal data!)
├── notebooks/              # Data exploration and model building
├── src/
│   ├── data/               # Data loaders/preprocessors
│   ├── models/             # ML/AI models (HRV, mood, crisis)
│   ├── ai/                 # Conversational AI code
│   ├── mobile/             # Mobile app / Wear OS integration
│   └── utils/              # Helpers and configuration files
├── tests/                  # Unit and integration tests
├── scripts/                # Train/evaluate/serve scripts
└── deployment/             # Docker, cloud config, release materials


## Getting Started

1. **Clone the repository**
    ```
    git clone https://github.com/hayn404/predictive-mental-health-smartwatch.git
    ```

2. **Install dependencies**
    ```
    pip install -r requirements.txt
    ```
3. **Explore the notebooks**
    - See `notebooks/` for data exploration and modeling examples.

4. **Run the main application or API**
    ```
    python scripts/run_app.py
    ```

## Dataset

- Uses the open-access **Heartbeat Orbits** dataset from Nature Scientific Data (2025) for model development and demonstration.
- [DOI: 10.1038/s41597-025-05801-3](https://www.nature.com/articles/s41597-025-05801-3)

## Team

- **Haneen Alaa**
- **Kareem Mohamed**
- **Mariam Zakary**
- **Youssef Mahmoud**

**Supervisor:** Prof. Khaled Mostafa El Sayed (Zewail City University)

## Contributing

- Issues and improvements are welcome; submit via GitHub Issues or pull requests.

## License

- This project is released under the MIT License (see LICENSE).

## Acknowledgments

- Heartbeat Orbits open dataset (Nature Scientific Data, 2025)[https://www.nature.com/articles/s41597-025-05801-3]

---

> For academic use and further research only. Not intended for clinical diagnosis or life-critical interventions.
