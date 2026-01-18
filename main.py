import argparse
from pathlib import Path

from src.config import Config
from src.split_data import main as split_main
from src.train import train as train_main


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="store_true", help="Create train/val splits")
    parser.add_argument("--train", action="store_true", help="Train ViT model")
    parser.add_argument("--gui", action="store_true", help="Run GUI")
    args = parser.parse_args()

    cfg = Config()

    if args.split:
        split_main(cfg)

    if args.train:
        best_path = train_main(cfg)
        print(f"Best model: {best_path}")
        # GUI'nin baktığı yola da kopyalamak istersen:
        # (isteğe bağlı; şimdilik train zaten runs/vit_experiment_01/model.pt kaydediyor)

    if args.gui:
        from gui.app import main as gui_main
        gui_main()

    if not (args.split or args.train or args.gui):
        print("Örnek kullanım:")
        print("  python main.py --split")
        print("  python main.py --train")
        print("  python main.py --gui")


if __name__ == "__main__":
    run()
