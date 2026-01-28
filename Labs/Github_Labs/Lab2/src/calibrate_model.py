import argparse
import os
from joblib import dump, load
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate a trained classifier and store the result in a configurable location."
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        required=True,
        help="Timestamp associated with the trained model build.",
    )
    parser.add_argument(
        "--method",
        choices=["isotonic", "sigmoid"],
        default="isotonic",
        help="Calibration technique to apply to the base estimator.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory where the original (uncalibrated) model lives.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/calibrated",
        help="Target directory where the calibrated model should be saved.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_model_name = f"model_{args.timestamp}_dt_model.joblib"
    base_model_path = os.path.join(args.model_dir, base_model_name)

    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Base model not found at {base_model_path}")

    base_model = load(base_model_path)

    X, y = load_breast_cancer(return_X_y=True)

    calibrator = CalibratedClassifierCV(
        base_estimator=base_model,
        method=args.method,
        cv="prefit",
    )
    # Fit only the calibration layer; the estimator itself stays untouched.
    calibrator.fit(X, y)

    os.makedirs(args.output_dir, exist_ok=True)
    calibrated_name = f"calibrated_model_{args.timestamp}_{args.method}.joblib"
    calibrated_path = os.path.join(args.output_dir, calibrated_name)
    dump(calibrator, calibrated_path)

    print(f"Calibrated model saved to {calibrated_path}")


if __name__ == "__main__":
    main()
