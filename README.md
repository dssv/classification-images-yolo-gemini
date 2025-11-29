# Estudo Comparativo: Gemini API vs YOLOv8 Classification

Estudo comparativo entre **Instance Classification com Gemini API** versus **Modelo de Classificação Tradicional (YOLOv8)** para o problema de classificação de imagens Cat vs Dog.

## Objetivo

Comparar o desempenho de dois métodos de classificação de imagens:
1. **Gemini API** (Google Generative AI) - Modelo de linguagem multimodal
2. **YOLOv8 Classification** - Modelo tradicional de deep learning

## Dataset

**Cat vs Dog Image Classification Dataset**
- Fonte: [Kaggle](https://www.kaggle.com/datasets/sunilthite/cat-or-dog-image-classification/data)
- Total de imagens: ~27.500
- Classes: Cat (Gato) e Dog (Cachorro)

## Metodologia

### Estrutura do Experimento

- **30 simulações** independentes
- **30 imagens por classe** em cada simulação
- **Total: 1.800 imagens** (900 gatos + 900 cachorros)

### Estrutura de Diretórios

```
dataset_simulations/
├── Sim01/
│   ├── classe01/  (30 imagens de gatos)
│   └── classe02/  (30 imagens de cachorros)
├── Sim02/
│   ├── classe01/
│   └── classe02/
...
└── Sim30/
    ├── classe01/
    └── classe02/
```

### Métricas Avaliadas

Para cada simulação, são calculadas:
- **Acurácia** (Accuracy)
- **Precisão** (Precision)
- **Recall** (Recall)
- **F1-Score**

### Validação Estatística

- **Teste Pareado de Wilcoxon**
- Nível de significância: α = 0.05 (95% de confiança)

## Instalação

### Pré-requisitos

- Python 3.8+
- pip

### Setup do Ambiente

```bash
# Criar ambiente virtual
python3 -m venv venv

# Ativar ambiente virtual
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### Configuração da API do Gemini

1. Copie o arquivo `.env.example` para `.env`:
```bash
cp .env.example .env
```

2. Obtenha sua chave da API do Gemini em: https://makersuite.google.com/app/apikey

3. Edite o arquivo `.env` e adicione sua chave:
```
GEMINI_API_KEY=sua_chave_aqui
```

⚠️ **Importante**: O arquivo `.env` contém informações sensíveis e já está incluído no `.gitignore`. Nunca compartilhe sua chave de API publicamente.

## Uso

### Opção 1: Executar Experimento Completo (Recomendado)

```bash
python3 run_experiment.py
```

Este script executa todo o pipeline:
1. Organiza o dataset em 30 simulações
2. Classifica com Gemini API
3. Treina e avalia com YOLOv8
4. Gera análises e visualizações

⚠️ **Atenção**: Este processo pode levar várias horas!

### Opção 2: Executar Etapas Individualmente

#### 1. Organizar Dataset

```bash
python3 src/organize_dataset.py
```

#### 2. Classificação com Gemini API

```bash
python3 src/classify_gemini.py
```

Tempo estimado: 15-30 minutos

#### 3. Classificação com YOLOv8

```bash
python3 src/classify_yolov8.py
```

Tempo estimado: 1-3 horas (dependendo do hardware)

Parâmetros de treinamento:
- Modelo: YOLOv8n-cls (nano)
- Épocas: 10
- Tamanho da imagem: 224x224
- Batch size: 16

#### 4. Análise e Visualizações

```bash
python3 src/analyze_results.py
```

## Estrutura do Projeto

```
.
├── archive/                    # Dataset original do Kaggle
│   ├── Train/
│   └── Test/
├── dataset_simulations/        # Dataset organizado em 30 simulações
├── models/                     # Modelos treinados
│   └── yolov8/
├── results/                    # Resultados e análises
│   ├── plots/                 # Gráficos
│   ├── gemini_metrics.csv     # Métricas do Gemini
│   ├── yolov8_metrics.csv     # Métricas do YOLOv8
│   ├── wilcoxon_test_results.csv
│   └── summary_report.txt
├── src/
│   ├── organize_dataset.py    # Organiza dataset
│   ├── classify_gemini.py     # Classificação com Gemini
│   ├── classify_yolov8.py     # Classificação com YOLOv8
│   └── analyze_results.py     # Análise e visualizações
├── requirements.txt
├── run_experiment.py          # Script master
└── README.md
```

## Resultados

Após executar o experimento, os seguintes arquivos serão gerados:

### Métricas

- `results/gemini_metrics.csv` - Métricas de todas as 30 simulações (Gemini)
- `results/yolov8_metrics.csv` - Métricas de todas as 30 simulações (YOLOv8)

### Visualizações

- `results/plots/boxplot_comparison.png` - Boxplots comparativos das 4 métricas
- `results/plots/line_comparison.png` - Gráficos de linha (Acurácia e F1-Score)

### Análise Estatística

- `results/wilcoxon_test_results.csv` - Resultados do teste de Wilcoxon
- `results/summary_report.txt` - Relatório completo em texto

## Configurações

### YOLOv8

Parâmetros de treinamento em `src/classify_yolov8.py`:
```python
EPOCHS = 10
IMAGE_SIZE = 224
BATCH_SIZE = 16
MODEL_SIZE = "n"  # nano
```

## Dependências Principais

- `google-generativeai` - Gemini API
- `ultralytics` - YOLOv8
- `torch` - PyTorch
- `torchvision` - Visão computacional
- `scikit-learn` - Métricas de avaliação
- `pandas` - Manipulação de dados
- `matplotlib` - Visualizações
- `seaborn` - Gráficos estatísticos
- `scipy` - Teste de Wilcoxon

## Hardware Recomendado

- **CPU**: Multi-core moderno
- **RAM**: 8GB+
- **GPU**: Opcional (acelera treinamento YOLOv8)
  - CUDA para NVIDIA
  - MPS para Apple Silicon
- **Armazenamento**: 5GB+ livres

## Notas

- O Gemini API possui rate limiting, por isso há delays entre requisições
- O YOLOv8 pode usar GPU se disponível (CUDA ou MPS)
- Os modelos treinados são salvos em `models/yolov8/`
- Cada simulação do YOLOv8 treina um modelo independente

## Licença

Este projeto é destinado a fins acadêmicos e de pesquisa.

## Autor

Douglas Vasconcelos
Mestrado - 2025
