"""
Script para organizar o dataset Cat vs Dog em 30 simulações
Estrutura de saída:
    /Sim01
        /classe01 (Cat)
        /classe02 (Dog)
    /Sim02
        /classe01 (Cat)
        /classe02 (Dog)
    ...
    /Sim30
        /classe01 (Cat)
        /classe02 (Dog)
"""

import os
import shutil
import random
from pathlib import Path
from typing import List

# Configurações
NUM_SIMULATIONS = 30
IMAGES_PER_CLASS = 30  # 30 imagens por classe em cada simulação
TRAIN_DIR = "archive/Train"
OUTPUT_DIR = "dataset_simulations"
SEED = 42

def get_image_files(directory: str) -> List[str]:
    """Obtém lista de arquivos de imagem em um diretório"""
    path = Path(directory)
    image_files = list(path.glob("*.jpg")) + list(path.glob("*.png"))
    return [str(f) for f in image_files]

def create_simulation_structure(sim_num: int, base_dir: str):
    """Cria estrutura de diretórios para uma simulação"""
    sim_path = Path(base_dir) / f"Sim{sim_num:02d}"
    classe01_path = sim_path / "classe01"  # Cat
    classe02_path = sim_path / "classe02"  # Dog

    classe01_path.mkdir(parents=True, exist_ok=True)
    classe02_path.mkdir(parents=True, exist_ok=True)

    return str(classe01_path), str(classe02_path)

def copy_images(source_files: List[str], dest_dir: str, num_images: int):
    """Copia um número específico de imagens para o diretório de destino"""
    selected_files = source_files[:num_images]

    for file_path in selected_files:
        file_name = os.path.basename(file_path)
        dest_path = os.path.join(dest_dir, file_name)
        shutil.copy2(file_path, dest_path)

    return len(selected_files)

def organize_dataset():
    """Organiza o dataset em 30 simulações"""
    print("=" * 60)
    print("Organizando Dataset Cat vs Dog em 30 Simulações")
    print("=" * 60)

    # Define seed para reproducibilidade
    random.seed(SEED)

    # Obtém todas as imagens
    print("\n[1/4] Carregando imagens...")
    cat_images = get_image_files(os.path.join(TRAIN_DIR, "Cat"))
    dog_images = get_image_files(os.path.join(TRAIN_DIR, "Dog"))

    print(f"  - Gatos encontrados: {len(cat_images)}")
    print(f"  - Cachorros encontrados: {len(dog_images)}")

    # Verifica se há imagens suficientes
    total_needed = IMAGES_PER_CLASS * NUM_SIMULATIONS
    if len(cat_images) < total_needed:
        print(f"\n⚠️  AVISO: Não há gatos suficientes para {NUM_SIMULATIONS} simulações")
        print(f"    Necessário: {total_needed}")
        print(f"    Disponível: {len(cat_images)}")

    if len(dog_images) < total_needed:
        print(f"\n⚠️  AVISO: Não há cachorros suficientes para {NUM_SIMULATIONS} simulações")
        print(f"    Necessário: {total_needed}")
        print(f"    Disponível: {len(dog_images)}")

    # Embaralha as imagens
    print("\n[2/4] Embaralhando imagens...")
    random.shuffle(cat_images)
    random.shuffle(dog_images)

    # Cria diretório de saída
    print(f"\n[3/4] Criando estrutura de diretórios em '{OUTPUT_DIR}'...")
    if os.path.exists(OUTPUT_DIR):
        print(f"  - Removendo diretório existente...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    # Organiza em 30 simulações
    print(f"\n[4/4] Criando {NUM_SIMULATIONS} simulações...")

    for sim_num in range(1, NUM_SIMULATIONS + 1):
        print(f"\n  Simulação {sim_num:02d}:", end=" ")

        # Cria estrutura de diretórios
        classe01_dir, classe02_dir = create_simulation_structure(sim_num, OUTPUT_DIR)

        # Calcula índices para esta simulação
        start_idx = (sim_num - 1) * IMAGES_PER_CLASS
        end_idx = sim_num * IMAGES_PER_CLASS

        # Seleciona imagens para esta simulação
        cat_subset = cat_images[start_idx:end_idx]
        dog_subset = dog_images[start_idx:end_idx]

        # Copia imagens
        num_cats = copy_images(cat_subset, classe01_dir, IMAGES_PER_CLASS)
        num_dogs = copy_images(dog_subset, classe02_dir, IMAGES_PER_CLASS)

        print(f"✓ classe01={num_cats} gatos, classe02={num_dogs} cachorros")

    print("\n" + "=" * 60)
    print("✓ Organização concluída com sucesso!")
    print("=" * 60)
    print(f"\nEstrutura criada em: {OUTPUT_DIR}/")
    print(f"  - {NUM_SIMULATIONS} simulações")
    print(f"  - {IMAGES_PER_CLASS} imagens por classe em cada simulação")
    print(f"  - Total: {NUM_SIMULATIONS * IMAGES_PER_CLASS * 2} imagens")
    print(f"  - classe01 = Cat (Gato)")
    print(f"  - classe02 = Dog (Cachorro)")
    print("\n")

if __name__ == "__main__":
    organize_dataset()
