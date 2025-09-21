# export_weights.py
import sys
from pathlib import Path
from tensorflow.keras.models import load_model

# Try to load models that used a Lambda(preprocess_input) by registering it
def load_any_model(path: Path):
    # 1) Try plain load (clean models with no Lambda)
    try:
        return load_model(path, compile=False, safe_mode=False)
    except Exception as e1:
        # 2) Try MobileNetV2 preprocess
        try:
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mnet_preprocess
            return load_model(path, custom_objects={"preprocess_input": mnet_preprocess},
                              compile=False, safe_mode=False)
        except Exception as e2:
            # 3) Try EfficientNet preprocess (just in case)
            try:
                from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
                return load_model(path, custom_objects={"preprocess_input": eff_preprocess},
                                  compile=False, safe_mode=False)
            except Exception as e3:
                raise RuntimeError(
                    f"Failed to load model:\n1) {e1}\n2) {e2}\n3) {e3}"
                )

def main():
    project = Path(__file__).resolve().parent
    models_dir = project / "models"

    # Use CLI args if provided: python export_weights.py in_model out_weights
    in_model = Path(sys.argv[1]) if len(sys.argv) > 1 else models_dir / "fruit_mobilenetv2_half_clean.keras"
    out_weights = Path(sys.argv[2]) if len(sys.argv) > 2 else models_dir / "fruit_mnet128_half_weights.h5"

    if not in_model.exists():
        # Auto-find any .keras model under models/
        candidates = sorted(models_dir.glob("*.keras"))
        if not candidates:
            raise FileNotFoundError(f"No .keras model found in {models_dir}. Pass an input path explicitly.")
        print(f"Input model not found: {in_model}\nUsing first found: {candidates[0]}")
        in_model = candidates[0]

    print("Loading:", in_model)
    model = load_any_model(in_model)

    print("Saving weights to:", out_weights)
    try:
        model.save_weights(out_weights)  # h5 inferred by extension
    except TypeError:
        # Some environments need explicit format
        model.save_weights(out_weights, save_format="h5")

    print("Done. Weights file:", out_weights)

if __name__ == "__main__":
    main()