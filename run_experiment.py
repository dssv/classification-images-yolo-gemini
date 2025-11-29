"""
Script master para executar o experimento completo
Estudo Comparativo: Gemini API vs YOLOv8 Classification
"""

import sys
import subprocess
from pathlib import Path

def run_command(description: str, command: str):
    """
    Executa um comando e mostra o progresso
    """
    print("\n" + "="*70)
    print(f"➤ {description}")
    print("="*70)

    result = subprocess.run(command, shell=True)

    if result.returncode != 0:
        print(f"\n❌ Erro ao executar: {description}")
        sys.exit(1)

    print(f"✓ {description} - Concluído!")

def main():
    """
    Executa todo o pipeline do experimento
    """
    print("="*70)
    print("EXPERIMENTO COMPLETO: GEMINI API vs YOLOv8")
    print("="*70)
    print("\nEste script irá executar:")
    print("1. Organização do dataset em 30 simulações")
    print("2. Classificação com Gemini API (30 simulações)")
    print("3. Treinamento e avaliação com YOLOv8 (30 simulações)")
    print("4. Análise comparativa e visualizações")
    print("\n⚠️  ATENÇÃO: Este processo pode levar várias horas!")
    print("\nPressione CTRL+C para cancelar ou ENTER para continuar...")

    try:
        input()
    except KeyboardInterrupt:
        print("\n\nExecução cancelada pelo usuário.")
        sys.exit(0)

    # Ativa ambiente virtual
    venv_python = "source venv/bin/activate && python3"

    # Passo 1: Organizar dataset
    run_command(
        "Passo 1/4: Organizando dataset em 30 simulações",
        f"{venv_python} src/organize_dataset.py"
    )

    # Passo 2: Classificação com Gemini
    print("\n" + "="*70)
    print("Passo 2/4: Classificação com Gemini API")
    print("="*70)
    print("\n⚠️  Este passo pode levar 15-30 minutos...")
    print("Pressione ENTER para continuar ou CTRL+C para pular...")

    try:
        input()
        run_command(
            "Executando classificação com Gemini API",
            f"{venv_python} src/classify_gemini.py"
        )
    except KeyboardInterrupt:
        print("\n⚠️  Passo 2 pulado. Certifique-se de ter os resultados do Gemini antes da análise!")

    # Passo 3: Treinamento com YOLOv8
    print("\n" + "="*70)
    print("Passo 3/4: Treinamento e avaliação com YOLOv8")
    print("="*70)
    print("\n⚠️  Este passo pode levar 1-3 horas dependendo do hardware...")
    print("Pressione ENTER para continuar ou CTRL+C para pular...")

    try:
        input()
        run_command(
            "Executando treinamento e avaliação com YOLOv8",
            f"{venv_python} src/classify_yolov8.py"
        )
    except KeyboardInterrupt:
        print("\n⚠️  Passo 3 pulado. Certifique-se de ter os resultados do YOLOv8 antes da análise!")

    # Passo 4: Análise e visualizações
    print("\n" + "="*70)
    print("Passo 4/4: Análise comparativa e visualizações")
    print("="*70)

    # Verifica se os arquivos de resultados existem
    results_dir = Path("results")
    gemini_csv = results_dir / "gemini_metrics.csv"
    yolov8_csv = results_dir / "yolov8_metrics.csv"

    if not gemini_csv.exists() or not yolov8_csv.exists():
        print("\n⚠️  AVISO: Resultados do Gemini ou YOLOv8 não encontrados!")
        print("Execute os passos 2 e 3 antes de continuar com a análise.")
        print("\nDeseja continuar mesmo assim? (y/n): ", end="")
        response = input().lower()
        if response != 'y':
            print("\nExecução cancelada.")
            sys.exit(0)

    run_command(
        "Gerando análise comparativa e visualizações",
        f"{venv_python} src/analyze_results.py"
    )

    # Resumo final
    print("\n" + "="*70)
    print("✓ EXPERIMENTO CONCLUÍDO COM SUCESSO!")
    print("="*70)
    print("\nResultados disponíveis em:")
    print("  📊 Gráficos: results/plots/")
    print("  📈 Métricas: results/")
    print("  🤖 Modelos YOLOv8: models/yolov8/")
    print("\nArquivos principais:")
    print("  • gemini_metrics.csv - Métricas do Gemini")
    print("  • yolov8_metrics.csv - Métricas do YOLOv8")
    print("  • wilcoxon_test_results.csv - Teste estatístico")
    print("  • summary_report.txt - Relatório completo")
    print("  • plots/boxplot_comparison.png - Boxplots")
    print("  • plots/line_comparison.png - Gráficos de linha")
    print("\n")

if __name__ == "__main__":
    main()
