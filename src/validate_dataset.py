"""
Script para validar e remover imagens corrompidas do dataset
"""

import cv2
from pathlib import Path
from PIL import Image

def validate_image(image_path: Path) -> bool:
    """
    Verifica se uma imagem pode ser lida corretamente
    """
    try:
        # Tenta com OpenCV
        img = cv2.imread(str(image_path))
        if img is None:
            return False

        # Tenta com PIL
        with Image.open(image_path) as pil_img:
            pil_img.verify()

        return True
    except Exception:
        return False

def clean_directory(directory: Path, dry_run: bool = True):
    """
    Remove imagens corrompidas de um diretório
    """
    print(f"\nVerificando: {directory}")

    corrupted = []
    total = 0

    for img_file in directory.glob("*.jpg"):
        total += 1
        if not validate_image(img_file):
            corrupted.append(img_file)
            print(f"  ❌ Corrompida: {img_file.name}")

    print(f"  Total: {total} imagens")
    print(f"  Corrompidas: {len(corrupted)} imagens")

    if corrupted and not dry_run:
        print(f"\n  Removendo {len(corrupted)} imagens...")
        for img_file in corrupted:
            img_file.unlink()
        print(f"  ✓ Imagens removidas")

    return len(corrupted)

def main():
    """
    Função principal
    """
    print("="*60)
    print("Validação do Dataset Cat vs Dog")
    print("="*60)

    train_cat = Path("archive/Train/Cat")
    train_dog = Path("archive/Train/Dog")
    test_cat = Path("archive/Test/Cat")
    test_dog = Path("archive/Test/Dog")

    total_corrupted = 0

    # Valida todos os diretórios
    print("\n[1/2] Escaneando imagens...")
    for directory in [train_cat, train_dog, test_cat, test_dog]:
        if directory.exists():
            corrupted = clean_directory(directory, dry_run=False)
            total_corrupted += corrupted

    print(f"\n{'='*60}")
    if total_corrupted > 0:
        print(f"✓ {total_corrupted} imagens corrompidas foram removidas")
    else:
        print("✓ Nenhuma imagem corrompida encontrada")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
