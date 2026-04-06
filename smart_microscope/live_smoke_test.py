from pathlib import Path
import sys

from live_inference import LivePredictor
from paths_config import OUTPUTS_DIR


def main():
    if len(sys.argv) < 2:
        print("Usage: python live_smoke_test.py /path/to/image.jpg [model_name]")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    model_name = sys.argv[2] if len(sys.argv) > 2 else "resnet"

    predictor = LivePredictor(model_name=model_name, threshold=0.5)
    result = predictor.predict_image(image_path)

    out_csv = OUTPUTS_DIR / "smoke_test_predictions.csv"
    predictor.append_to_csv(result, out_csv)

    print("Prediction result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print(f"Saved CSV row to: {out_csv}")


if __name__ == "__main__":
    main()
