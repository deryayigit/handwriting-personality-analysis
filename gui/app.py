if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QSizePolicy
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

from gui.controller import AppController


class HandwritingApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("El Yazısından Kişilik Analizi")
        self.resize(800, 400)

        self.image_path = None
        self.model_path = "runs/vit_experiment_01/model.pt"

        self.controller = AppController(self, self.model_path)

        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout()
        content_layout = QHBoxLayout()

        # Sol kutu: Yazı Önizleme
        left_box = QGroupBox("Yazı Önizleme")
        left_layout = QVBoxLayout()

        self.image_label = QLabel("Görsel Seçilmedi")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedHeight(300)

        self.select_btn = QPushButton("Görsel Seçiniz")
        self.select_btn.clicked.connect(self.select_image)

        left_layout.addWidget(self.image_label)
        left_layout.addWidget(self.select_btn)
        left_box.setLayout(left_layout)

        # Sağ kutu: Kişilik Özellikleri
        right_box = QGroupBox("Kişilik Özellikleri")
        trait_layout = QVBoxLayout()

        self.trait_names = [
            "Uyumluluk (Agreeableness)",
            "Özdisiplin (Conscientiousness)",
            "Dışadönüklük (Extraversion)",
            "Duygusal Dengesizlik (Neuroticism)",
            "Açık Fikirlilik (Openness)"
        ]

        self.trait_value_labels = []

        for name in self.trait_names:
            row = QHBoxLayout()
            label_name = QLabel(name)
            label_value = QLabel("—")
            label_value.setAlignment(Qt.AlignmentFlag.AlignRight)

            self.trait_value_labels.append(label_value)

            row.addWidget(label_name)
            row.addWidget(label_value)
            trait_layout.addLayout(row)

        right_box.setLayout(trait_layout)

        # conf_box = QGroupBox("Kimlik Doğrulama")
        # conf_layout = QVBoxLayout()
        #
        # self.model_conf_label = QLabel("%—")
        # self.entropy_conf_label = QLabel("%—")
        #
        # conf_layout.addWidget(QLabel("Model Güven Skoru"))
        # conf_layout.addWidget(self.model_conf_label)
        #
        # conf_layout.addWidget(QLabel("Entropy Güven"))
        # conf_layout.addWidget(self.entropy_conf_label)
        #
        # conf_box.setLayout(conf_layout)

        # İçeriği yatay sıraya ekle
        content_layout.addWidget(left_box, 1)
        content_layout.addWidget(right_box, 1)

        # Analiz butonu (tam genişlik)
        self.analyze_btn = QPushButton("Analiz Ediniz")
        self.analyze_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.analyze_btn.clicked.connect(self.run_analysis)

        main_layout.addLayout(content_layout)
        # main_layout.addWidget(conf_box)  # (Eğer arayüzde yeniden görünmesi istenirse aç)
        main_layout.addWidget(self.analyze_btn)

        self.setLayout(main_layout)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Görsel Seçiniz", "", "Images (*.png *.jpg *.jpeg)"
        )

        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(file_path).scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio
            )
            self.image_label.setPixmap(pixmap)

    def run_analysis(self):
        if not self.image_path:
            return
        self.controller.analyze(self.image_path)

    def update_traits(self, traits):
        for label, value in zip(self.trait_value_labels, traits):
            label.setText(f"{value:.1f}")

    # Fonksiyon içeride çalışmaya devam etsin ama arayüzde gösterilmesin
    def update_confidence(self, model_conf, entropy_conf):
        # self.model_conf_label.setText(f"%{model_conf:.1f}")
        # self.entropy_conf_label.setText(f"%{entropy_conf:.1f}")
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandwritingApp()
    window.show()
    sys.exit(app.exec())
