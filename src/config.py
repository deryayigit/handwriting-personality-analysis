from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    
    data_root: Path = Path("data/raw/kaggle_personality/training_set")
    runs_dir: Path = Path("runs")
    split_file: Path = Path("runs/splits.json")

    class_names = (
        "Agreeableness",
        "Conscientiousness",
        "Extraversion",
        "Neuroticism",
        "Openness",
    )

    seed: int = 42
    val_ratio: float = 0.2
    
    experiment_name: str = "vit_experiment_01"

    model_name: str = "vit_base_patch16_224"
    num_classes: int = 5
    img_size: int = 224
    dropout: float = 0.1

    batch_size: int = 16
    num_workers: int = 0
    epochs_head: int = 3
    epochs_finetune: int = 8
    lr_head: float = 3e-4
    lr_finetune: float = 5e-5
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    use_amp: bool = False
