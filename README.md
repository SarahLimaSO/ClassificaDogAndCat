# Classificação Binária Neuro-Simbólica: Gatos vs. Cachorros com LTNtorch

Este projeto implementa um classificador binário de imagens (Gato vs. Cachorro) utilizando uma abordagem **Neuro-Simbólica** baseada no framework **Logic Tensor Networks (LTN)**.

Diferente das abordagens tradicionais de Deep Learning que minimizam o erro entre um rótulo e uma predição, este modelo aprende **maximizando a satisfação de regras lógicas** definidas em uma Base de Conhecimento.
---
### integrantes
* Antonio Lucas
* Breno
* Caio
* Lucas da Silva Moura
* Luiz Felipe Nery Soares
* Sarah Campos Fernandes Lima
* Rafael Emanuel Dantas Viana
* Victor José Nunes Kossman
---
## 🧠 O Conceito: Logic Tensor Networks (LTN)

O LTN integra o aprendizado profundo (Redes Neurais) com o raciocínio lógico (Lógica de Primeira Ordem Fuzzy). O processo se baseia em três pilares principais:

1.  **Lógica Real (Real Logic):** Uma linguagem onde os símbolos lógicos são interpretados como tensores (dados) e funções diferenciáveis (redes neurais).
2.  **Grounding (Aterramento/Ancoragem):** O mapeamento dos dados reais para os símbolos lógicos. [cite_start]Por exemplo, conectar um conjunto de imagens à variável lógica $x$.
3.  **Aprendizado via Satisfação:** O treinamento busca ajustar os pesos da rede neural para que as fórmulas lógicas da base de conhecimento sejam verdadeiras (valor de verdade próximo de 1).

## 📋 O Problema e a Modelagem Lógica

**Objetivo:** Classificar corretamente se uma imagem é de um **Cachorro** ou de um **Gato** usando o dataset CIFAR-10.

### A Base de Conhecimento ($\mathcal{K}$)

Definimos um predicado $Dog(x)$ que representa uma Rede Neural (CNN).Esta rede recebe uma imagem $x$ e retorna a probabilidade (grau de verdade) de ser um cachorro.

O modelo é treinado para satisfazer dois axiomas lógicos fundamentais:

1.  **Axioma Positivo:** "Para toda imagem de cachorro ($x_{dog}$), o predicado $Dog$ deve ser verdadeiro."
    $$\forall x_{dog} (Dog(x_{dog}))$$

2.  **Axioma Negativo:** "Para toda imagem de gato ($x_{cat}$), o predicado $Dog$ **NÃO** deve ser verdadeiro."
    $$\forall x_{cat} (\neg Dog(x_{cat}))$$

### Função de Perda (Loss)

A função de perda não compara rótulos diretamente. Ela é derivada da satisfação agregada da base de conhecimento ($SatAgg$):

$$\mathcal{L} = 1 - SatAgg(\mathcal{K})$$

O otimizador trabalha para minimizar essa perda, o que equivale a maximizar a verdade das regras lógicas.

## 🛠️ Arquitetura e Implementação

[cite_start]O código está estruturado nas seguintes etapas, conforme proposto na documentação do LTNtorch[cite: 19, 189]:

### 1. Preparação dos Dados
* **Dataset:** CIFAR-10.
* **Filtragem:** Seleciona-se apenas as classes índice 3 (Gatos) e 5 (Cachorros).
* **Normalização:** Imagens convertidas para tensores normalizados.
* **Separação:** Os dados são divididos em dois grupos (`cats_data` e `dogs_data`) para permitir o *grounding* correto das variáveis lógicas.

### 2. O Predicado (Rede Neural)
Uma **CNN Simples** é utilizada como a "inteligência" por trás do predicado $Dog$.
* **Entrada:** Imagens 32x32 pixels (3 canais).
* **Estrutura:** 2 camadas convolucionais + Max Pooling + 3 camadas lineares.
* **Saída:** Um único neurônio com ativação **Sigmoid**, garantindo um valor de verdade no intervalo fuzzy $[0, 1]$.

### 3. Operadores Fuzzy
O LTN substitui operadores booleanos por operadores difusos diferenciáveis:
* **Conectivo NOT ($\neg$):** Negação padrão ($1 - x$).
* **Quantificador FORALL ($\forall$):** Agregador baseado em erro médio (*p-mean error*).

### 4. Loop de Treinamento
A cada época:
1.  Amostra-se um batch de cães e um de gatos.
2.  **Grounding:** Cria-se variáveis LTN (`var_dog`, `var_cat`) associadas às imagens.
3. **Avaliação:** As fórmulas $\forall x_{dog} Dog(x)$ e $\forall x_{cat} \neg Dog(x)$ são calculadas.
4. **Backpropagation:** O gradiente flui através da estrutura lógica até os pesos da CNN para maximizar a satisfação.

## 🚀 Como Executar

### Pré-requisitos

```bash
pip install torch torchvision ltntorch matplotlib numpy
```
### Exemplo
```python
# Exemplo de saída esperada após o treino:
# Predicado Dog(imagem_cachorro) = 0.9991 (Esperado: ~1.0)
# Predicado Dog(imagem_gato)     = 0.0277 (Esperado: ~0.0)
