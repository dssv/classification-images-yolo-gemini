"""
Script para identificar imagens duplicadas entre Train e Test
Verifica duplicatas por:
1. Nome do arquivo
2. Hash do conteúdo (mais confiável)
"""

import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List, Tuple

# Configurações
TRAIN_DIR = "archive/Train"
TEST_DIR = "archive/Test"

def calculate_file_hash(file_path: Path) -> str:
    """
    Calcula o hash MD5 de um arquivo
    """
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"  ⚠️  Erro ao calcular hash de {file_path.name}: {e}")
        return None

def get_files_with_hashes(directory: Path, class_name: str) -> Dict[str, Path]:
    """
    Obtém todos os arquivos de imagem com seus hashes
    Retorna: {hash: file_path}
    """
    print(f"\n  Processando {directory}...")

    files_dict = {}
    total = 0

    for img_file in directory.glob("*.jpg"):
        total += 1
        file_hash = calculate_file_hash(img_file)
        if file_hash:
            files_dict[file_hash] = img_file

        if total % 1000 == 0:
            print(f"    Processadas: {total} imagens", end="\r")

    print(f"    Processadas: {total} imagens ✓")
    return files_dict

def check_duplicates_by_hash(train_files: Dict[str, Path], test_files: Dict[str, Path]) -> List[Tuple[Path, Path]]:
    """
    Identifica duplicatas por hash de conteúdo
    """
    duplicates = []

    train_hashes = set(train_files.keys())
    test_hashes = set(test_files.keys())

    common_hashes = train_hashes & test_hashes

    for hash_val in common_hashes:
        duplicates.append((train_files[hash_val], test_files[hash_val]))

    return duplicates

def check_duplicates_by_name(train_dir: Path, test_dir: Path) -> List[Tuple[str, str]]:
    """
    Identifica duplicatas por nome de arquivo
    """
    train_names = {f.name for f in train_dir.glob("*.jpg")}
    test_names = {f.name for f in test_dir.glob("*.jpg")}

    common_names = train_names & test_names

    duplicates = []
    for name in common_names:
        duplicates.append((str(train_dir / name), str(test_dir / name)))

    return duplicates

def remove_duplicates_from_test(duplicates_by_hash: Dict[str, List]) -> int:
    """
    Remove imagens duplicadas do conjunto de teste
    Retorna o número de "imagens removidas"
    """
    total_removed = 0

    for class_name, duplicates in duplicates_by_hash.items():
        print(f"\n  Removendo duplicatas da classe {class_name}...")

        for train_file, test_file in duplicates:
            try:
                test_file.unlink()  # Remove o arquivo
                total_removed += 1
                print(f"    ✓ Removida: {test_file.name}")
            except Exception as e:
                print(f"    ❌ Erro ao remover {test_file.name}: {e}")

    return total_removed

def save_duplicates_report(duplicates_by_hash: Dict[str, List],
                          duplicates_by_name: Dict[str, List],
                          output_file: str,
                          removed_count: int = 0):
    """
    Salva relatório de duplicatas
    """
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RELATÓRIO DE IMAGENS DUPLICADAS\n")
        f.write("Train vs Test Dataset\n")
        f.write("="*80 + "\n\n")

        # Duplicatas por hash
        f.write("1. DUPLICATAS POR CONTEÚDO (Hash MD5)\n")
        f.write("-"*80 + "\n\n")

        total_hash_duplicates = sum(len(dups) for dups in duplicates_by_hash.values())

        if total_hash_duplicates > 0:
            f.write(f"Total de duplicatas encontradas: {total_hash_duplicates}\n\n")

            for class_name, duplicates in duplicates_by_hash.items():
                if duplicates:
                    f.write(f"\nClasse: {class_name}\n")
                    f.write(f"Duplicatas: {len(duplicates)}\n\n")

                    for train_file, test_file in duplicates:
                        f.write(f"  Train: {train_file}\n")
                        f.write(f"  Test:  {test_file}\n")
                        f.write("\n")
        else:
            f.write("✓ Nenhuma duplicata por conteúdo encontrada!\n")

        # Duplicatas por nome
        f.write("\n\n2. DUPLICATAS POR NOME DE ARQUIVO\n")
        f.write("-"*80 + "\n\n")

        total_name_duplicates = sum(len(dups) for dups in duplicates_by_name.values())

        if total_name_duplicates > 0:
            f.write(f"Total de duplicatas encontradas: {total_name_duplicates}\n\n")

            for class_name, duplicates in duplicates_by_name.items():
                if duplicates:
                    f.write(f"\nClasse: {class_name}\n")
                    f.write(f"Duplicatas: {len(duplicates)}\n\n")

                    for train_file, test_file in duplicates:
                        train_name = Path(train_file).name
                        test_name = Path(test_file).name
                        f.write(f"  Nome: {train_name}\n")
                        f.write(f"  Train: {train_file}\n")
                        f.write(f"  Test:  {test_file}\n")
                        f.write("\n")
        else:
            f.write("✓ Nenhuma duplicata por nome encontrada!\n")

        # Resumo
        f.write("\n\n3. RESUMO\n")
        f.write("-"*80 + "\n\n")
        f.write(f"Total de duplicatas por conteúdo: {total_hash_duplicates}\n")
        f.write(f"Total de duplicatas por nome: {total_name_duplicates}\n\n")

        if total_hash_duplicates > 0 or total_name_duplicates > 0:
            f.write("⚠️  ATENÇÃO: Foram encontradas imagens duplicadas!\n")
            f.write("Isso pode comprometer a validade do experimento.\n")
            f.write("Recomendação: Remover duplicatas do conjunto de teste.\n")
        else:
            f.write("✓ Dataset está limpo! Nenhuma duplicata encontrada.\n")

def main():
    """
    Função principal
    """
    print("="*80)
    print("Verificação de Imagens Duplicadas: Train vs Test")
    print("="*80)

    # Paths
    train_cat = Path(TRAIN_DIR) / "Cat"
    train_dog = Path(TRAIN_DIR) / "Dog"
    test_cat = Path(TEST_DIR) / "Cat"
    test_dog = Path(TEST_DIR) / "Dog"

    # Verifica se diretórios existem
    for directory in [train_cat, train_dog, test_cat, test_dog]:
        if not directory.exists():
            print(f"❌ Erro: Diretório não encontrado: {directory}")
            return

    # 1. Verificação por HASH (mais confiável)
    print("\n[1/2] Verificando duplicatas por conteúdo (Hash MD5)...")

    duplicates_by_hash = {}

    # Classe Cat
    print("\n  Classe: Cat")
    train_cat_files = get_files_with_hashes(train_cat, "Cat")
    test_cat_files = get_files_with_hashes(test_cat, "Cat")
    cat_duplicates = check_duplicates_by_hash(train_cat_files, test_cat_files)
    duplicates_by_hash["Cat"] = cat_duplicates

    print(f"\n  → Duplicatas encontradas (Cat): {len(cat_duplicates)}")

    # Classe Dog
    print("\n  Classe: Dog")
    train_dog_files = get_files_with_hashes(train_dog, "Dog")
    test_dog_files = get_files_with_hashes(test_dog, "Dog")
    dog_duplicates = check_duplicates_by_hash(train_dog_files, test_dog_files)
    duplicates_by_hash["Dog"] = dog_duplicates

    print(f"\n  → Duplicatas encontradas (Dog): {len(dog_duplicates)}")

    # 2. Verificação por NOME (rápida, mas pode dar falsos positivos)
    print("\n[2/2] Verificando duplicatas por nome de arquivo...")

    duplicates_by_name = {}

    cat_name_duplicates = check_duplicates_by_name(train_cat, test_cat)
    duplicates_by_name["Cat"] = cat_name_duplicates
    print(f"  → Duplicatas por nome (Cat): {len(cat_name_duplicates)}")

    dog_name_duplicates = check_duplicates_by_name(train_dog, test_dog)
    duplicates_by_name["Dog"] = dog_name_duplicates
    print(f"  → Duplicatas por nome (Dog): {len(dog_name_duplicates)}")

    # Totais
    total_hash = len(cat_duplicates) + len(dog_duplicates)
    total_name = len(cat_name_duplicates) + len(dog_name_duplicates)

    print(f"\n{'='*80}")
    print("RESUMO")
    print(f"{'='*80}")
    print(f"Total de duplicatas por CONTEÚDO: {total_hash}")
    print(f"Total de duplicatas por NOME:     {total_name}")

    # Salva relatório
    output_file = "results/duplicates_report.txt"
    Path("results").mkdir(exist_ok=True)
    save_duplicates_report(duplicates_by_hash, duplicates_by_name, output_file)
    print(f"\n✓ Relatório salvo em: {output_file}")

    # Mostra algumas duplicatas se houver
    if total_hash > 0:
        print(f"\n{'='*80}")
        print("⚠️  ATENÇÃO: DUPLICATAS ENCONTRADAS!")
        print(f"{'='*80}")
        print("\nExemplos de duplicatas (primeiras 5):")

        count = 0
        for class_name, duplicates in duplicates_by_hash.items():
            for train_file, test_file in duplicates[:5]:
                count += 1
                print(f"\n{count}. Classe: {class_name}")
                print(f"   Train: {train_file.name}")
                print(f"   Test:  {test_file.name}")
                if count >= 5:
                    break
            if count >= 5:
                break

        print(f"\n⚠️  Isso pode comprometer a validade do experimento!")
        print(f"⚠️  Recomendação: Remover duplicatas do conjunto de teste.")

        # Remove duplicadas
        remove_duplicates_from_test(duplicates_by_hash)
    else:
        print(f"\n{'='*80}")
        print("✓ DATASET LIMPO!")
        print(f"{'='*80}")
        print("Nenhuma duplicata encontrada entre Train e Test.")
        print("O dataset está adequado para experimentação.")

    print(f"\n{'='*80}")



if __name__ == "__main__":
    main()
