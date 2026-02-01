import argparse
from pathlib import Path
import joblib
import glob


def find_best_model(project_root: Path) -> Path:
    """
    Encuentra automáticamente un modelo guardado en news/runs/tfidf/best_tfidf_*.joblib
    y devuelve el más reciente por fecha de modificación.
    """
    tfidf_dir = project_root / "news" / "runs" / "tfidf"
    pattern = str(tfidf_dir / "best_tfidf_*.joblib")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(
            f"No encontré modelos en {tfidf_dir}. Ejecuta antes: python main.py"
        )
    candidates.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return Path(candidates[0])


def predict_text(model, text: str) -> str:
    pred = model.predict([text])[0]
    return "FAKE ❌" if int(pred) == 1 else "REAL ✅"


def main():
    parser = argparse.ArgumentParser(description="Predicción manual de noticias (Fake/Real)")
    parser.add_argument("--model", type=str, default=None, help="Ruta al modelo .joblib (opcional)")
    parser.add_argument("--text", type=str, default=None, help="Texto de noticia para clasificar")
    parser.add_argument("--file", type=str, default=None, help="Ruta a .txt con la noticia")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent

    model_path = Path(args.model) if args.model else find_best_model(project_root)
    print("Usando modelo:", model_path)

    model = joblib.load(model_path)

    # Caso 1: texto por argumento
    if args.text:
        print("Predicción:", predict_text(model, args.text))
        return

    # Caso 2: texto desde fichero
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            raise FileNotFoundError(f"No existe el archivo: {file_path}")
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        print("Predicción:", predict_text(model, text))
        return

    # Caso 3: modo interactivo
    print("\nModo interactivo. Pega una noticia (Enter para salir):\n")
    while True:
        txt = input("> ").strip()
        if not txt:
            break
        print("Predicción:", predict_text(model, txt))


if __name__ == "__main__":
    main()
