"""
Script para avaliação de imagens Cat vs Dog usando modelo YOLOv8 já treinado
Carrega o modelo treinado e avalia nas 30 simulações
"""

import os
from pathlib import Path
from typing import Dict, List
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import json

# Configurações
SIMULATIONS_DIR = "dataset_simulations"
OUTPUT_DIR = "results"
MODELS_DIR = "models/yolov8"
NUM_SIMULATIONS = 30
RESULTS_FILE = "yolov8_results.json"
MODEL_NAME = "cat_dog_classifier"

def load_trained_model() -> YOLO:
    """
    Carrega o modelo YOLOv8 já treinado
    """
    print("\n[1/2] Carregando modelo YOLOv8 treinado...")

    # Caminho para o melhor modelo
    best_model_path = Path(MODELS_DIR) / MODEL_NAME / "weights" / "best.pt"

    if not best_model_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em: {best_model_path}\n"
            f"Execute o treinamento primeiro com: python3 src/classify_yolov8.py"
        )

    model = YOLO(str(best_model_path))
    print(f"  ✓ Modelo carregado de: {best_model_path}")

    return model

def evaluate_simulation(model: YOLO, sim_num: int) -> Dict:
    """
    Avalia o modelo em uma simulação específica
    """
    sim_path = Path(SIMULATIONS_DIR) / f"Sim{sim_num:02d}"

    y_true = []
    y_pred = []

    # Processa classe01 (Cat)
    classe01_path = sim_path / "classe01"
    for img_file in classe01_path.glob("*.jpg"):
        results = model(str(img_file), verbose=False)

        probs = results[0].probs
        predicted_class_id = probs.top1
        predicted_class_name = results[0].names[predicted_class_id]

        y_true.append("Cat")
        y_pred.append(predicted_class_name)

    # Processa classe02 (Dog)
    classe02_path = sim_path / "classe02"
    for img_file in classe02_path.glob("*.jpg"):
        results = model(str(img_file), verbose=False)

        probs = results[0].probs
        predicted_class_id = probs.top1
        predicted_class_name = results[0].names[predicted_class_id]

        y_true.append("Dog")
        y_pred.append(predicted_class_name)

    # Calcula métricas
    metrics = calculate_metrics(y_true, y_pred)

    return metrics

def calculate_metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    """
    Calcula as métricas de avaliação
    """
    # Converte labels para binário (Cat=0, Dog=1)
    label_map = {"Cat": 0, "Dog": 1}
    y_true_bin = [label_map.get(label, 0) for label in y_true]
    y_pred_bin = [label_map.get(label, 0) for label in y_pred]

    accuracy = accuracy_score(y_true_bin, y_pred_bin)
    precision = precision_score(y_true_bin, y_pred_bin, average='weighted', zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, average='weighted', zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, average='weighted', zero_division=0)

    # Matriz de confusão
    cm = confusion_matrix(y_true_bin, y_pred_bin)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist()
    }

def evaluate_all_simulations(model: YOLO) -> List[Dict]:
    """
    Avalia o modelo em todas as 30 simulações
    """
    print(f"\n[2/2] Avaliando modelo nas {NUM_SIMULATIONS} simulações...")

    all_results = []

    for sim_num in range(1, NUM_SIMULATIONS + 1):
        print(f"  Simulação {sim_num:02d}...", end=" ")

        metrics = evaluate_simulation(model, sim_num)

        result = {
            "simulation": sim_num,
            "metrics": metrics
        }

        all_results.append(result)

        print(f"Acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")

    return all_results

def save_results(all_results: List[Dict]):
    """
    Salva os resultados em JSON e CSV
    """
    print(f"\n[3/3] Salvando resultados...")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Salva JSON
    json_path = output_dir / RESULTS_FILE
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  ✓ JSON salvo em: {json_path}")

    # Cria e salva CSV
    data = []
    for result in all_results:
        data.append({
            "Simulação": result["simulation"],
            "Acurácia": result["metrics"]["accuracy"],
            "Precision": result["metrics"]["precision"],
            "Recall": result["metrics"]["recall"],
            "F1-Score": result["metrics"]["f1_score"]
        })

    df = pd.DataFrame(data)
    csv_path = output_dir / "yolov8_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✓ CSV salvo em: {csv_path}")

    # Mostra estatísticas
    print(f"\n{'='*60}")
    print("Estatísticas Descritivas - YOLOv8")
    print(f"{'='*60}")
    print(df.describe())

def main():
    """
    Função principal
    """
    print("="*60)
    print("Avaliação com YOLOv8 - Cat vs Dog")
    print("="*60)
    print(f"Simulações para avaliar: {NUM_SIMULATIONS}")

    # Carrega modelo treinado
    model = load_trained_model()

    # Avalia nas 30 simulações
    all_results = evaluate_all_simulations(model)

    # Salva resultados
    save_results(all_results)

    print(f"\n{'='*60}")
    print("✓ Avaliação concluída com sucesso!")
    print(f"{'='*60}")
    print(f"\nResultados salvos em: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
