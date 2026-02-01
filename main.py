import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_curve,
    roc_auc_score,
)

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import joblib



# CONFIG (main.py en RAÍZ, datos en news/)
RANDOM_SEED = 42

ROOT_DIR = Path(__file__).resolve().parent
NEWS_DIR = ROOT_DIR / "news"

FAKE_PATH = NEWS_DIR / "Fake.csv"
TRUE_PATH = NEWS_DIR / "True.csv"

RUNS_DIR = NEWS_DIR / "runs"
RUNS_DIR.mkdir(exist_ok=True)



# NLT
def ensure_nltk():
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")

    try:
        nltk.corpus.wordnet.ensure_loaded()
    except LookupError:
        nltk.download("wordnet")


ensure_nltk()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# 1) CARGA DATASET
def load_dataset(fake_path: Path, true_path: Path, seed: int = 42) -> pd.DataFrame:
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 1  # fake
    true_df["label"] = 0  # real

    df = pd.concat([fake_df, true_df], axis=0, ignore_index=True)

    if "text" not in df.columns:
        raise ValueError("No existe la columna 'text' en el dataset.")

    df["text"] = df["text"].fillna("").astype(str)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# 2) LEAKAGE CHECK: Reuters
def leakage_check_reuters(df: pd.DataFrame):
    tag_regex = r"\(\s*reuters\s*\)"
    df["has_reuters_tag"] = df["text"].str.contains(tag_regex, case=False, regex=True)

    print("\n[Leakage check] % con '(Reuters)' por clase (0=Real, 1=Fake):")
    print(df.groupby("label")["has_reuters_tag"].mean().round(4))

    print("\n[Leakage check] Conteo '(Reuters)' por clase:")
    print(df.groupby("label")["has_reuters_tag"].sum())

    # ejemplos rápidos para evidenciarlo
    print("\n[Leakage check] Ejemplos (primeros 2 REAL con Reuters):")
    ex_real = df[(df["label"] == 0) & (df["has_reuters_tag"])].head(2)["text"].tolist()
    for i, t in enumerate(ex_real, 1):
        print(f"  REAL ex {i}: {t[:120]}...")

    print("\n[Leakage check] Ejemplos (primeros 2 FAKE con Reuters):")
    ex_fake = df[(df["label"] == 1) & (df["has_reuters_tag"])].head(2)["text"].tolist()
    if not ex_fake:
        print("  (No encontré ejemplos FAKE con '(Reuters)' en las primeras filas filtradas.)")
    else:
        for i, t in enumerate(ex_fake, 1):
            print(f"  FAKE ex {i}: {t[:120]}...")


# 3) PREPROCESAMIENTO (eliminación del tag Reuters)
def remove_reuters_words(text: str) -> str:
    return re.sub(r"\S*reuter\S*", " ", str(text), flags=re.IGNORECASE)


def remove_reuters_anywhere(text: str) -> str:
    """
    Elimina cualquier token que contenga 'reuter' en cualquier posición,
    sin distinguir mayúsculas/minúsculas.
    Cubre: Reuters, (Reuters), REUTERS:, washington (Reuters) -, reuters.com,
    y también 'reuter' (por lematización de 'reuters').
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    # \S*reuter\S* => borra el token completo donde aparezca reuter
    text = re.sub(r"(?i)\S*reuter\S*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text(text: str, remove_reuters: bool = True) -> str:
    if remove_reuters:
        text = remove_reuters_anywhere(text)
        text = remove_reuters_words(text)

    text = text.lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # seguridad extra: por si queda algo tras normalización
    text = re.sub(r"\breuter[s]?\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

# 4) EDA
def run_eda(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[EDA] Distribución de labels:")
    print(df["label"].value_counts())

    plt.figure()
    sns.countplot(x="label", data=df)
    plt.title("Distribución de Labels (0=Real, 1=Fake)")
    plt.tight_layout()
    plt.savefig(out_dir / "label_distribution.png", dpi=150)
    plt.close()

    df["char_len"] = df["text"].apply(len)
    df["word_len"] = df["text"].apply(lambda x: len(x.split()))
    df[["char_len", "word_len"]].describe().to_csv(out_dir / "length_stats.csv")

    plt.figure()
    sns.histplot(data=df, x="word_len", hue="label", bins=50)
    plt.title("Longitud en palabras por clase")
    plt.tight_layout()
    plt.savefig(out_dir / "word_len_hist.png", dpi=150)
    plt.close()

    plt.figure()
    sns.histplot(data=df, x="char_len", hue="label", bins=50)
    plt.title("Longitud en caracteres por clase")
    plt.tight_layout()
    plt.savefig(out_dir / "char_len_hist.png", dpi=150)
    plt.close()

    print(f"[EDA] Guardado en: {out_dir}")


def save_confusion_matrix(y_true, y_pred, out_path: Path, title: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cbar=False)
    plt.title(title)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_roc_curve_if_possible(model, X_test, y_test, out_path: Path):
    if not hasattr(model, "predict_proba"):
        return None

    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    fpr, tpr, _ = roc_curve(y_test, probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[EXTRA] ROC guardada en: {out_path} (AUC={auc:.4f})")
    return float(auc)


# 5) TF-IDF + CLASIFICACIÓN
def train_tfidf(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    X = df["clean_text"].astype(str).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "LinearSVM": LinearSVC(),
    }

    results = {}
    best_name, best_f1, best_pipe = None, -1, None

    for name, clf in models.items():
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=10000)),
            ("clf", clf),
        ])

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        f1 = f1_score(y_test, pred)
        acc = accuracy_score(y_test, pred)

        print(f"\n[TF-IDF] Modelo: {name} | Acc={acc:.4f} | F1={f1:.4f}")
        save_confusion_matrix(
            y_test, pred,
            out_dir / f"confusion_{name}.png",
            title=f"Confusion Matrix - TFIDF {name}"
        )

        auc = None
        if name == "LogisticRegression":
            auc = save_roc_curve_if_possible(pipe, X_test, y_test, out_dir / "roc_curve.png")

        results[name] = {
            "accuracy": float(acc),
            "f1": float(f1),
            "auc": auc,
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "report": classification_report(y_test, pred, output_dict=True),
        }

        if f1 > best_f1:
            best_f1, best_name, best_pipe = f1, name, pipe

    with open(out_dir / "tfidf_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    best_model_path = out_dir / f"best_tfidf_{best_name}.joblib"
    joblib.dump(best_pipe, best_model_path)

    print(f"\nMejor TF-IDF: {best_name} con F1={best_f1:.4f}")
    print(f"Modelo guardado en: {best_model_path}")

    return best_name, float(best_f1), best_model_path



# 6) EMBEDDINGS + CLASIFICACIÓN
def embed_texts(model, texts, batch_size=64):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, show_progress_bar=False)
        embeddings.append(emb)
    return np.vstack(embeddings)


def train_embeddings(df: pd.DataFrame, out_dir: Path, sample_n: int | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)

    df_use = df
    if sample_n is not None and sample_n > 0 and sample_n < len(df):
        df_use = df.sample(n=sample_n, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"[Embeddings] Usando muestra de {sample_n} textos para acelerar.")

    X = df_use["clean_text"].astype(str).tolist()
    y = df_use["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    st_model = SentenceTransformer("all-MiniLM-L12-v2")

    X_train_emb = embed_texts(st_model, X_train, batch_size=64)
    X_test_emb = embed_texts(st_model, X_test, batch_size=64)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train_emb, y_train)

    pred = clf.predict(X_test_emb)
    f1 = f1_score(y_test, pred)
    acc = accuracy_score(y_test, pred)

    print(f"\n[Embeddings] Acc={acc:.4f} | F1={f1:.4f}")

    save_confusion_matrix(
        y_test, pred,
        out_dir / "confusion_embeddings.png",
        title="Confusion Matrix - Embeddings"
    )

    results = {
        "model": "all-MiniLM-L12-v2",
        "accuracy": float(acc),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "report": classification_report(y_test, pred, output_dict=True),
        "sample_n": sample_n,
    }

    with open(out_dir / "embeddings_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Guardar clasificador embeddings
    joblib.dump(clf, out_dir / "embeddings_classifier.joblib")
    # Guardar el nombre del modelo para inferencia futura
    joblib.dump("all-MiniLM-L12-v2", out_dir / "embeddings_st_model_name.joblib")

    return float(f1)



# 7) Comparativa final
def save_final_comparison(best_tfidf_name, best_tfidf_f1, embeddings_f1, out_path: Path):
    rows = [{"method": f"TF-IDF ({best_tfidf_name})", "f1": best_tfidf_f1}]
    if embeddings_f1 is not None:
        rows.append({"method": "Embeddings (MiniLM) + LogReg", "f1": embeddings_f1})

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[EXTRA] Comparativa final guardada en: {out_path}")


# Main
def main():
    print("=== Fake News Detector (Reuters leakage fixed) ===")

    if not FAKE_PATH.exists() or not TRUE_PATH.exists():
        raise FileNotFoundError(
            f"No encuentro CSV. Esperaba:\n- {FAKE_PATH}\n- {TRUE_PATH}"
        )

    # 1) Cargar
    print("\n[1] Cargando dataset...")
    df = load_dataset(FAKE_PATH, TRUE_PATH, seed=RANDOM_SEED)
    print("Shape:", df.shape)

    # 2) Leakage check Reuters
    leakage_check_reuters(df)

    # 3) EDA
    print("\n[2] EDA...")
    run_eda(df, RUNS_DIR / "eda")

    # 4) Preprocesado (se elimina Reuters)
    print("\n[3] Preprocesando (eliminando '(Reuters)' para evitar leakage)...")
    df["clean_text"] = df["text"].apply(lambda t: clean_text(t, remove_reuters=True))

    out_csv = RUNS_DIR / "processed_dataset.csv"
    try:
        df.to_csv(out_csv, index=False)
        print(f"[OK] Guardado: {out_csv}")
    except PermissionError:
        alt = RUNS_DIR / "processed_dataset_tmp.csv"
        df.to_csv(alt, index=False)
        print(f"[WARN] No pude escribir {out_csv} (¿archivo abierto?). Guardé en: {alt}")

    # 5) TF-IDF
    print("\n[4] TF-IDF training...")
    tfidf_dir = RUNS_DIR / "tfidf"
    best_name, best_f1, best_model_path = train_tfidf(df, tfidf_dir)

    # 6) Embeddings
    # Para acelerar se puede poner un sample_n=10000 por ejemplo
    print("\n[5] Embeddings training (opcional)...")
    embeddings_f1 = train_embeddings(df, RUNS_DIR / "embeddings", sample_n=10000)

    # 7) Comparativa
    save_final_comparison(best_name, best_f1, embeddings_f1, RUNS_DIR / "final_comparison.csv")

    print("\nFIN. Revisa outputs en: news/runs/")
    print("Modelo TF-IDF listo para usar en predict_manual.py:", best_model_path)


if __name__ == "__main__":
    main()
