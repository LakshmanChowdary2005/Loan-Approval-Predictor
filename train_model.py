"""Compatibility entry point for the shared loan model training pipeline."""

from src.train_model import train_and_evaluate


def run_training():
    return train_and_evaluate()


if __name__ == "__main__":
    run_training()
