"""
Script para classificação de imagens Cat vs Dog usando Gemini API
Processa todas as 30 simulações e calcula métricas de desempenho
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import google.generativeai as genai
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Configurações
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY não encontrada no arquivo .env")

SIMULATIONS_DIR = "dataset_simulations"
OUTPUT_DIR = "results"
NUM_SIMULATIONS = 30
RESULTS_FILE = "gemini_results.json"

# Mapeamento de classes
CLASS_MAPPING = {
    "classe01": "cat",  # Gato
    "classe02": "dog"   # Cachorro
}

def setup_gemini():
    """Configura a API do Gemini"""
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
    return model

def classify_image(model, image_path: str) -> str:
    """
    Classifica uma imagem usando Gemini API
    Retorna: 'cat' ou 'dog'
    """
    try:
        # Carrega a imagem
        img = Image.open(image_path)

        # Prompt para classificação
        prompt = """
        Analyze this image and determine if it contains a cat or a dog.
        Respond with ONLY ONE WORD: either "cat" or "dog".
        Do not include any other text, explanations, or punctuation in your response.
        """

        # Gera a resposta
        response = model.generate_content([prompt, img])

        # Processa a resposta
        prediction = response.text.strip().lower()

        # Garante que a resposta seja válida
        if "cat" in prediction:
            return "cat"
        elif "dog" in prediction:
            return "dog"
        else:
            # Se a resposta for ambígua, tenta novamente com prompt mais direto
            prompt_simple = "Is this a cat or a dog? Answer with only: cat or dog"
            response = model.generate_content([prompt_simple, img])
            prediction = response.text.strip().lower()

            if "cat" in prediction:
                return "cat"
            elif "dog" in prediction:
                return "dog"
            else:
                print(f"  ⚠️  Resposta ambígua: {prediction}, assumindo 'dog'")
                return "dog"

    except Exception as e:
        print(f"  ❌ Erro ao classificar {image_path}: {e}")
        return "dog"  # Valor padrão em caso de erro

def process_simulation(model, sim_num: int) -> Dict:
    """
    Processa uma simulação completa e retorna as métricas
    """
    print(f"\n{'='*60}")
    print(f"Processando Simulação {sim_num:02d}")
    print(f"{'='*60}")

    sim_path = Path(SIMULATIONS_DIR) / f"Sim{sim_num:02d}"

    y_true = []
    y_pred = []

    # Processa cada classe
    for class_dir, true_label in CLASS_MAPPING.items():
        class_path = sim_path / class_dir
        image_files = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png"))

        print(f"\n  Classificando {class_dir} ({true_label}):")
        print(f"  Total de imagens: {len(image_files)}")

        for i, image_path in enumerate(image_files, 1):
            # Classifica a imagem
            prediction = classify_image(model, str(image_path))

            # Armazena os resultados
            y_true.append(true_label)
            y_pred.append(prediction)

            # Mostra progresso
            if i % 10 == 0 or i == len(image_files):
                print(f"    Processadas: {i}/{len(image_files)}", end="\r")

            # Pequeno delay para evitar rate limiting
            time.sleep(0.1)

        print(f"    Processadas: {len(image_files)}/{len(image_files)} ✓")

    # Calcula métricas
    metrics = calculate_metrics(y_true, y_pred)

    print(f"\n  Resultados da Simulação {sim_num:02d}:")
    print(f"  - Acurácia:  {metrics['accuracy']:.4f}")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall:    {metrics['recall']:.4f}")
    print(f"  - F1-Score:  {metrics['f1_score']:.4f}")

    return {
        "simulation": sim_num,
        "metrics": metrics,
        "predictions": {
            "y_true": y_true,
            "y_pred": y_pred
        }
    }

def calculate_metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    """
    Calcula as métricas de avaliação
    """
    # Converte labels para binário (cat=0, dog=1)
    y_true_bin = [0 if label == "cat" else 1 for label in y_true]
    y_pred_bin = [0 if label == "cat" else 1 for label in y_pred]

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

def save_results(all_results: List[Dict], output_file: str):
    """
    Salva os resultados em JSON
    """
    # Remove predictions detalhadas para economizar espaço
    results_summary = []
    for result in all_results:
        results_summary.append({
            "simulation": result["simulation"],
            "metrics": result["metrics"]
        })

    output_path = Path(OUTPUT_DIR) / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n✓ Resultados salvos em: {output_path}")

def create_summary_csv(all_results: List[Dict]):
    """
    Cria um CSV com resumo das métricas
    """
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

    # Salva CSV
    csv_path = Path(OUTPUT_DIR) / "gemini_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Métricas salvas em CSV: {csv_path}")

    # Mostra estatísticas descritivas
    print(f"\n{'='*60}")
    print("Estatísticas Descritivas - Gemini")
    print(f"{'='*60}")
    print(df.describe())

    return df

def main():
    """
    Função principal
    """
    print("="*60)
    print("Classificação com Gemini API - Cat vs Dog")
    print("="*60)
    print(f"Total de simulações: {NUM_SIMULATIONS}")
    print(f"Modelo: gemini-1.5-flash")

    # Configura o modelo
    print("\n[1/3] Configurando Gemini API...")
    model = setup_gemini()
    print("✓ Gemini configurado com sucesso!")

    # Processa todas as simulações
    print(f"\n[2/3] Processando {NUM_SIMULATIONS} simulações...")
    all_results = []

    start_time = time.time()

    for sim_num in range(1, NUM_SIMULATIONS + 1):
        result = process_simulation(model, sim_num)
        all_results.append(result)

    elapsed_time = time.time() - start_time

    # Salva resultados
    print(f"\n[3/3] Salvando resultados...")
    save_results(all_results, RESULTS_FILE)
    create_summary_csv(all_results)

    print(f"\n{'='*60}")
    print("✓ Processamento concluído com sucesso!")
    print(f"{'='*60}")
    print(f"Tempo total: {elapsed_time/60:.2f} minutos")
    print(f"Tempo médio por simulação: {elapsed_time/NUM_SIMULATIONS:.2f} segundos")
    print(f"\nResultados salvos em: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
