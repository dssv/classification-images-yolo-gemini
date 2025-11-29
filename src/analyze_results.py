"""
Script para análise comparativa entre Gemini API e YOLOv8
Gera visualizações avançadas e testes estatísticos
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List

# Configurações
OUTPUT_DIR = "results"
PLOTS_DIR = "results/plots"
GEMINI_RESULTS = "gemini_results.json"
YOLOV8_RESULTS = "yolov8_results.json"
GEMINI_METRICS_CSV = "gemini_metrics.csv"
YOLOV8_METRICS_CSV = "yolov8_metrics.csv"

# Configuração de estilo
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega os resultados do Gemini e YOLOv8
    """
    print("\n[1/6] Carregando resultados...")

    output_path = Path(OUTPUT_DIR)

    # Carrega CSVs
    gemini_df = pd.read_csv(output_path / GEMINI_METRICS_CSV)
    yolov8_df = pd.read_csv(output_path / YOLOV8_METRICS_CSV)

    print(f"  ✓ Gemini: {len(gemini_df)} simulações")
    print(f"  ✓ YOLOv8: {len(yolov8_df)} simulações")

    return gemini_df, yolov8_df

def create_comparison_dataframe(gemini_df: pd.DataFrame, yolov8_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria DataFrame combinado para comparação
    """
    # Reorganiza dados para formato long
    gemini_long = gemini_df.melt(
        id_vars=['Simulação'],
        value_vars=['Acurácia', 'Precision', 'Recall', 'F1-Score'],
        var_name='Métrica',
        value_name='Valor'
    )
    gemini_long['Modelo'] = 'Gemini API'

    yolov8_long = yolov8_df.melt(
        id_vars=['Simulação'],
        value_vars=['Acurácia', 'Precision', 'Recall', 'F1-Score'],
        var_name='Métrica',
        value_name='Valor'
    )
    yolov8_long['Modelo'] = 'YOLOv8'

    combined_df = pd.concat([gemini_long, yolov8_long], ignore_index=True)

    return combined_df

def plot_boxplot_comparison(combined_df: pd.DataFrame, plots_dir: Path):
    """
    Cria boxplots comparativos para cada métrica
    """
    print("\n[2/6] Gerando boxplots comparativos...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comparação de Métricas: Gemini API vs YOLOv8', fontsize=16, fontweight='bold')

    metrics = ['Acurácia', 'Precision', 'Recall', 'F1-Score']
    colors = ['#3498db', '#e74c3c']

    for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
        data = combined_df[combined_df['Métrica'] == metric]

        sns.boxplot(data=data, x='Modelo', y='Valor', ax=ax, palette=colors)
        sns.stripplot(data=data, x='Modelo', y='Valor', ax=ax,
                     color='black', alpha=0.3, size=3)

        ax.set_title(f'{metric}', fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Valor', fontsize=12)
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)

        # Adiciona médias
        for i, model in enumerate(['Gemini API', 'YOLOv8']):
            mean_val = data[data['Modelo'] == model]['Valor'].mean()
            ax.hlines(mean_val, i-0.2, i+0.2, colors='red', linestyles='--', linewidth=2, label='Média')
            ax.text(i, mean_val+0.02, f'{mean_val:.4f}', ha='center', fontweight='bold')

    plt.tight_layout()
    output_path = plots_dir / 'boxplot_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Boxplots salvos: {output_path}")

def plot_line_comparison(gemini_df: pd.DataFrame, yolov8_df: pd.DataFrame, plots_dir: Path):
    """
    Cria gráficos de linha mostrando evolução por simulação
    """
    print("\n[3/6] Gerando gráficos de linha...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Evolução das Métricas por Simulação', fontsize=16, fontweight='bold')

    metrics = ['Acurácia', 'Precision', 'Recall', 'F1-Score']

    for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
        ax.plot(gemini_df['Simulação'], gemini_df[metric],
               marker='o', label='Gemini API', linewidth=2, markersize=4, color='#3498db')
        ax.plot(yolov8_df['Simulação'], yolov8_df[metric],
               marker='s', label='YOLOv8', linewidth=2, markersize=4, color='#e74c3c')

        ax.set_title(f'{metric}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Simulação', fontsize=12)
        ax.set_ylabel('Valor', fontsize=12)
        ax.set_ylim([0, 1.05])
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = plots_dir / 'line_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Gráficos de linha salvos: {output_path}")

def plot_radar_chart(gemini_df: pd.DataFrame, yolov8_df: pd.DataFrame, plots_dir: Path):
    """
    Cria gráfico de radar comparando médias das métricas
    """
    print("\n[4/6] Gerando gráfico de radar...")

    # Calcula médias
    metrics = ['Acurácia', 'Precision', 'Recall', 'F1-Score']
    gemini_means = [gemini_df[m].mean() for m in metrics]
    yolov8_means = [yolov8_df[m].mean() for m in metrics]

    # Configuração do radar
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    gemini_means += gemini_means[:1]  # Fecha o círculo
    yolov8_means += yolov8_means[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Plot
    ax.plot(angles, gemini_means, 'o-', linewidth=2, label='Gemini API', color='#3498db')
    ax.fill(angles, gemini_means, alpha=0.25, color='#3498db')

    ax.plot(angles, yolov8_means, 's-', linewidth=2, label='YOLOv8', color='#e74c3c')
    ax.fill(angles, yolov8_means, alpha=0.25, color='#e74c3c')

    # Configurações
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, size=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=10)
    ax.grid(True, alpha=0.3)

    # Título e legenda
    plt.title('Comparação Média das Métricas\nGemini API vs YOLOv8',
             size=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)

    plt.tight_layout()
    output_path = plots_dir / 'radar_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Gráfico de radar salvo: {output_path}")

def plot_violin_comparison(combined_df: pd.DataFrame, plots_dir: Path):
    """
    Cria violin plots mostrando distribuição das métricas
    """
    print("\n[5/6] Gerando violin plots...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Distribuição das Métricas: Gemini API vs YOLOv8', fontsize=16, fontweight='bold')

    metrics = ['Acurácia', 'Precision', 'Recall', 'F1-Score']
    colors = ['#3498db', '#e74c3c']

    for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
        data = combined_df[combined_df['Métrica'] == metric]

        sns.violinplot(data=data, x='Modelo', y='Valor', ax=ax, palette=colors)

        ax.set_title(f'{metric}', fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Valor', fontsize=12)
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = plots_dir / 'violin_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Violin plots salvos: {output_path}")

def plot_heatmap(gemini_df: pd.DataFrame, yolov8_df: pd.DataFrame, plots_dir: Path):
    """
    Cria heatmap das diferenças entre modelos por simulação
    """
    print("\n[6/6] Gerando heatmap de diferenças...")

    metrics = ['Acurácia', 'Precision', 'Recall', 'F1-Score']

    # Calcula diferenças (YOLOv8 - Gemini)
    diff_data = {}
    for metric in metrics:
        diff_data[metric] = yolov8_df[metric].values - gemini_df[metric].values

    diff_df = pd.DataFrame(diff_data, index=gemini_df['Simulação'])

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(diff_df.T, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
               cbar_kws={'label': 'Diferença (YOLOv8 - Gemini)'}, ax=ax,
               linewidths=0.5, linecolor='gray')

    ax.set_title('Diferença de Desempenho por Simulação\n(Valores Positivos = YOLOv8 Melhor)',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Simulação', fontsize=12)
    ax.set_ylabel('Métrica', fontsize=12)

    plt.tight_layout()
    output_path = plots_dir / 'heatmap_differences.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Heatmap salvo: {output_path}")

def perform_statistical_tests(gemini_df: pd.DataFrame, yolov8_df: pd.DataFrame, output_dir: Path):
    """
    Realiza testes estatísticos (Wilcoxon paired test)
    """
    print("\n[7/7] Executando testes estatísticos...")

    metrics = ['Acurácia', 'Precision', 'Recall', 'F1-Score']
    results = []

    for metric in metrics:
        gemini_values = gemini_df[metric].values
        yolov8_values = yolov8_df[metric].values

        # Teste de Wilcoxon (pareado)
        statistic, p_value = stats.wilcoxon(gemini_values, yolov8_values)

        # Calcula estatísticas descritivas
        gemini_mean = gemini_values.mean()
        yolov8_mean = yolov8_values.mean()
        diff_mean = yolov8_mean - gemini_mean

        # Significância
        is_significant = p_value < 0.05

        results.append({
            'Métrica': metric,
            'Gemini_Média': gemini_mean,
            'YOLOv8_Média': yolov8_mean,
            'Diferença': diff_mean,
            'Estatística_Wilcoxon': statistic,
            'P-valor': p_value,
            'Significativo_5%': is_significant
        })

    # Salva resultados
    results_df = pd.DataFrame(results)
    csv_path = output_dir / 'wilcoxon_test_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"  ✓ Resultados salvos: {csv_path}")

    # Gera relatório em texto
    report_path = output_dir / 'summary_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RELATÓRIO DE ANÁLISE COMPARATIVA\n")
        f.write("Gemini API vs YOLOv8 Classification\n")
        f.write("="*80 + "\n\n")

        f.write("1. ESTATÍSTICAS DESCRITIVAS\n")
        f.write("-"*80 + "\n\n")

        for _, row in results_df.iterrows():
            f.write(f"Métrica: {row['Métrica']}\n")
            f.write(f"  Gemini API - Média: {row['Gemini_Média']:.4f}\n")
            f.write(f"  YOLOv8     - Média: {row['YOLOv8_Média']:.4f}\n")
            f.write(f"  Diferença:          {row['Diferença']:+.4f}\n")
            f.write(f"  Melhor modelo:      {'YOLOv8' if row['Diferença'] > 0 else 'Gemini API'}\n\n")

        f.write("\n2. TESTE DE WILCOXON (Pareado)\n")
        f.write("-"*80 + "\n\n")

        for _, row in results_df.iterrows():
            f.write(f"Métrica: {row['Métrica']}\n")
            f.write(f"  Estatística: {row['Estatística_Wilcoxon']:.4f}\n")
            f.write(f"  P-valor:     {row['P-valor']:.6f}\n")
            f.write(f"  Significativo (α=0.05): {'SIM' if row['Significativo_5%'] else 'NÃO'}\n")

            if row['Significativo_5%']:
                melhor = 'YOLOv8' if row['Diferença'] > 0 else 'Gemini API'
                f.write(f"  → Há diferença significativa a favor de {melhor}\n")
            else:
                f.write(f"  → Não há diferença estatisticamente significativa\n")
            f.write("\n")

        f.write("\n3. CONCLUSÃO\n")
        f.write("-"*80 + "\n\n")

        sig_count = results_df['Significativo_5%'].sum()
        if sig_count > 0:
            f.write(f"Foram encontradas diferenças significativas em {sig_count} de 4 métricas.\n")
        else:
            f.write("Não foram encontradas diferenças estatisticamente significativas.\n")

    print(f"  ✓ Relatório salvo: {report_path}")

    # Imprime resumo
    print(f"\n{'='*80}")
    print("RESUMO DOS TESTES ESTATÍSTICOS")
    print(f"{'='*80}")
    print(results_df.to_string(index=False))

def main():
    """
    Função principal
    """
    print("="*80)
    print("Análise Comparativa: Gemini API vs YOLOv8")
    print("="*80)

    # Cria diretório de plots
    plots_dir = Path(PLOTS_DIR)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Carrega resultados
    gemini_df, yolov8_df = load_results()

    # Cria DataFrame combinado
    combined_df = create_comparison_dataframe(gemini_df, yolov8_df)

    # Gera visualizações
    plot_boxplot_comparison(combined_df, plots_dir)
    plot_line_comparison(gemini_df, yolov8_df, plots_dir)
    plot_radar_chart(gemini_df, yolov8_df, plots_dir)
    plot_violin_comparison(combined_df, plots_dir)
    plot_heatmap(gemini_df, yolov8_df, plots_dir)

    # Testes estatísticos
    output_dir = Path(OUTPUT_DIR)
    perform_statistical_tests(gemini_df, yolov8_df, output_dir)

    print(f"\n{'='*80}")
    print("✓ Análise concluída com sucesso!")
    print(f"{'='*80}")
    print(f"\nVisualizações salvas em: {plots_dir}/")
    print(f"  - boxplot_comparison.png     - Boxplots comparativos")
    print(f"  - line_comparison.png        - Evolução por simulação")
    print(f"  - radar_comparison.png       - Gráfico de radar")
    print(f"  - violin_comparison.png      - Distribuição das métricas")
    print(f"  - heatmap_differences.png    - Diferenças por simulação")
    print(f"\nRelatórios salvos em: {output_dir}/")
    print(f"  - wilcoxon_test_results.csv  - Resultados dos testes")
    print(f"  - summary_report.txt         - Relatório completo")

if __name__ == "__main__":
    main()
