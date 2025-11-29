Projeto: Estudo comparatico entre Instance Classification com Gemini vs. Modelo de Classificação

Definição das atividades:

1. Preparar Dataser de Validação

a. Exemplo:

/Sim01
    /classe01
    /classe02
/Sim02
    /classe01
    /classe02
...

2. Requisitos do Dataset

a. Pelo menos 3000 Imagens da classe 01 e classe 02

3. Experimento e métricas 

a. Para cada simulação extrair *Acurácia, Precision, Recall, F1-Score*

b. Executar 30 simulações:

b1. Plotar um gráfico de boxplot para cada métrica acima, comparando as abordagens.
b2. Gráfico de linha comparando acurácia e F1-ScribdDocument

4. Validação

Teste pareado de Wilcoxon com P=0.05 (Grau de confiança de 95%)

Dataset: https://www.kaggle.com/datasets/sunilthite/cat-or-dog-image-classification/data 

Cat and Dog classification: Identificar se na imagem recebida encontramos gato ou cachorro.