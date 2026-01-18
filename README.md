# Handwriting-Based Personality Analysis using Vision Transformers

Handwriting is a complex human behavior that goes beyond the mere transcription of text onto paper, containing various clues about an individual’s cognitive processes, motor skills, and psychological state. The traces formed from the moment the pen comes into contact with the paper are not simply the result of a physical movement, but rather the outward manifestation of a multilayered process directed by the brain. In this context, handwriting can be considered a unique biometric indicator that reflects an individual’s characteristic traits. Each person’s handwriting is distinctive; similar to fingerprints or DNA, it constitutes a form of biometric data that is specific to the individual. This dynamic structure may also contain variable information related to a person’s emotional state and inner condition. Therefore, handwriting is regarded not only as a means of identity verification, but also as a meaningful data source for character and personality analysis.

The relationship between handwriting and personality analysis is examined within the scope of the science of **Graphology**. Graphology is a field of study that aims to derive inferences about an individual’s personality and character traits based on their handwriting. This approach has found applications in various disciplines such as forensic investigations, human resources processes, psychological counseling, and education.

In this study, an **engineering design** aimed at analyzing personality traits from individuals’ handwriting samples has been implemented using deep learning methods. Instead of classical graphological feature extraction techniques, the **Vision Transformer (ViT)** architecture has been preferred, and the processes of feature extraction and classification have been modeled using an end-to-end learning approach. Personality prediction is performed based on the **Big Five Personality Model (Big Five Personality Traits – OCEAN)**.

---

## System Architecture

The system is designed in a modular and extensible manner:

* **src/**: Core deep learning pipeline (training, evaluation, inference)
* **gui/**: PyQt6-based graphical user interface
* **main.py**: Central entry point for training, evaluation, and GUI execution

The graphical user interface and the deep learning model are decoupled, enabling independent development and potential future integration with web or mobile-based interfaces.

---

## Dataset

This project uses the **Personality Prediction using Handwriting Images** dataset available on Kaggle.
Due to licensing and ethical considerations, the dataset is **not included** in this repository.

The dataset consists of handwriting images labeled according to the **Big Five Personality Traits**.
These labels are treated as *reference personality categories* and are **not intended as clinical or definitive psychological diagnoses**.

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the GUI:

```bash
python -m gui.app
```

(Optional) Train the model:

```bash
python main.py --train
```

---

## Limitations

* The dataset size is limited, which may affect model generalization.
* Personality labels are derived from graphology-based annotations rather than clinical assessments.
* The system should be considered a **decision-support tool**, not a diagnostic system.

---

## Author & Academic Context

**Derya Yiğit**
Computer Engineering
Karadeniz Technical University

This project was developed as part of the *Engineering Design and Final Year Project* during the 2025–2026 academic year.
