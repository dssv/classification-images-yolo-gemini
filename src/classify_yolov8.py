"""
Script para classificação de imagens Cat vs Dog usando YOLOv8 Classification
1. Treina um único modelo usando archive/Train
2. Usa esse modelo para classificar as 30 simulações
3. Calcula métricas para cada simulação
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List
import torch
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd

# Configurações
TRAIN_DIR = "archive/Train"
TEST_DIR = "archive/Test"
SIMULATIONS_DIR = "dataset_simulations"
OUTPUT_DIR = "results"
MODELS_DIR = "models/yolov8"
NUM_SIMULATIONS = 30
RESULTS_FILE = "yolov8_results.json"
MODEL_NAME = "cat_dog_classifier"

# Parâmetros de treinamento
EPOCHS = 10
IMAGE_SIZE = 224
BATCH_SIZE = 16
MODEL_SIZE = "n"  # nano, s, m, l, x - 'n' é o menor e mais rápido

def prepare_training_dataset() -> str:
    """
    Prepara o dataset de treinamento usando archive/Train e archive/Test
    YOLOv8 Classification espera estrutura: dataset/train/class_name/images
    """
    print("\n[1/4] Preparando dataset de treinamento...")

    yolo_dataset_path = Path("yolo_train_dataset")

    # Remove dataset anterior se existir
    if yolo_dataset_path.exists():
        shutil.rmtree(yolo_dataset_path)

    # Cria estrutura train/val
    train_path = yolo_dataset_path / "train"
    val_path = yolo_dataset_path / "val"

    # Cria diretórios
    for split_path in [train_path, val_path]:
        (split_path / "Cat").mkdir(parents=True, exist_ok=True)
        (split_path / "Dog").mkdir(parents=True, exist_ok=True)

    # Copia imagens de treino
    print("  Copiando imagens de treino...")
    for class_name in ["Cat", "Dog"]:
        src_dir = Path(TRAIN_DIR) / class_name
        dst_dir = train_path / class_name

        for img_file in src_dir.glob("*.jpg"):
            shutil.copy2(img_file, dst_dir / img_file.name)

    # Copia imagens de validação
    print("  Copiando imagens de validação...")
    for class_name in ["Cat", "Dog"]:
        src_dir = Path(TEST_DIR) / class_name
        dst_dir = val_path / class_name

        for img_file in src_dir.glob("*.jpg"):
            shutil.copy2(img_file, dst_dir / img_file.name)

    # Conta imagens
    train_cat = len(list((train_path / "Cat").glob("*.jpg")))
    train_dog = len(list((train_path / "Dog").glob("*.jpg")))
    val_cat = len(list((val_path / "Cat").glob("*.jpg")))
    val_dog = len(list((val_path / "Dog").glob("*.jpg")))

    print(f"  ✓ Train: {train_cat} gatos, {train_dog} cachorros")
    print(f"  ✓ Val:   {val_cat} gatos, {val_dog} cachorros")
    print(f"  ✓ Path: {yolo_dataset_path}")
    return str(yolo_dataset_path)

def train_model(dataset_path: str) -> YOLO:
    """
    Treina um modelo YOLOv8 Classification
    """
    print(f"\n[2/4] Treinando modelo YOLOv8 ({EPOCHS} épocas)...")

    # Inicializa o modelo
    model = YOLO(f'yolov8{MODEL_SIZE}-cls.pt')

    # Diretório para salvar modelo
    model_dir = Path(MODELS_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Treina o modelo
    results = model.train(
        data=dataset_path,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=str(model_dir),
        name=MODEL_NAME,
        exist_ok=True,
        verbose=True,
        plots=True,
        device='mps' if torch.backends.mps.is_available() else 'cpu'
    )

    # Carrega o melhor modelo treinado
    best_model_path = model_dir / MODEL_NAME / "weights" / "best.pt"
    model = YOLO(str(best_model_path))

    print(f"  ✓ Modelo treinado e salvo em: {best_model_path}")

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
    print(f"\n[3/4] Avaliando modelo nas {NUM_SIMULATIONS} simulações...")

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
    print(f"\n[4/4] Salvando resultados...")

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
    print("Classificação com YOLOv8 - Cat vs Dog")
    print("="*60)
    print(f"Modelo: YOLOv8{MODEL_SIZE}-cls")
    print(f"Épocas de treinamento: {EPOCHS}")
    print(f"Tamanho da imagem: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Simulações para avaliar: {NUM_SIMULATIONS}")

    # Verifica dispositivo
    if torch.backends.mps.is_available():
        print(f"Dispositivo: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        print(f"Dispositivo: CUDA (GPU)")
    else:
        print(f"Dispositivo: CPU")

    # Prepara dataset de treinamento
    # dataset_path = prepare_training_dataset()

    # Treina modelo
    model = train_model("archive")

    # Avalia nas 30 simulações
    all_results = evaluate_all_simulations(model)

    # Salva resultados
    save_results(all_results)

    # Limpa dataset temporário
    print("\n  Limpando arquivos temporários...")
    if Path(dataset_path).exists():
        shutil.rmtree(dataset_path)

    print(f"\n{'='*60}")
    print("✓ Processamento concluído com sucesso!")
    print(f"{'='*60}")
    print(f"\nResultados salvos em: {OUTPUT_DIR}/")
    print(f"Modelo salvo em: {MODELS_DIR}/{MODEL_NAME}/")

if __name__ == "__main__":
    main()
