"""
Script to train ML models.
"""
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.ml import ModelTrainer
from app.core.data import data_loader


def main():
    """Train all models."""
    print("="*60)
    print("Fraud Detection Model Training")
    print("="*60)
    
    # Check if data exists
    try:
        df = data_loader.load()
        print(f"\nDataset loaded: {len(df)} rows")
        print(f"Fraud cases: {df['Class'].sum()}")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease download the dataset from:")
        print("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        sys.exit(1)
    
    # Train models
    trainer = ModelTrainer()
    models, metrics = trainer.train_all(save=True)
    
    # Print comparison
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    comparison = trainer.get_model_comparison()
    print(comparison.to_string(index=False))
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)


if __name__ == "__main__":
    main()
