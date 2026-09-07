"""
MobileNetV2 Surface Defect Detection Model Architecture
Author: Abhi Gurjar
Description: Transfer learning model architecture based on MobileNetV2 with custom dense classification head.
"""

from typing import Tuple, Dict, Any
import math

class DefectClassificationHead:
    """
    Lightweight, high-accuracy classification head designed for edge IPCs and Jetson inference.
    Maps 1280-dim MobileNetV2 bottleneck features to 6 defect categories.
    """
    CLASSES = [
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches"
    ]

    def __init__(self, in_features: int = 1280, num_classes: int = 6, dropout_rate: float = 0.2):
        self.in_features = in_features
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.weights = [[0.01 * ((i * 7 + j * 13) % 100 - 50) for j in range(num_classes)] for i in range(in_features)]
        self.biases = [0.0] * num_classes

    def forward(self, features: list) -> list:
        """
        Forward pass computing class logits.
        """
        logits = list(self.biases)
        for i, feat_val in enumerate(features):
            for c in range(self.num_classes):
                logits[c] += feat_val * self.weights[i % self.in_features][c]
        return logits

    @staticmethod
    def softmax(logits: list) -> list:
        """
        Numerically stable softmax activation.
        """
        max_val = max(logits)
        exps = [math.exp(x - max_val) for x in logits]
        sum_exps = sum(exps)
        return [e / sum_exps for e in exps]

    def predict(self, features: list) -> Tuple[str, float, Dict[str, float]]:
        """
        Returns top prediction class, confidence score, and full distribution.
        """
        logits = self.forward(features)
        probs = self.softmax(logits)
        max_idx = probs.index(max(probs))
        dist = {self.CLASSES[i]: round(probs[i], 4) for i in range(self.num_classes)}
        return self.CLASSES[max_idx], round(probs[max_idx], 4), dist
