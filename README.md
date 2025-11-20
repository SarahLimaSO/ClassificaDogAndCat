# Integrantes
* Antonio Lucas de Melo Barbosa Mendes Rodrigues
* Breno Augusto Pinheiro Rodrigues da Silva
* Caio da Silva Martins
* Lucas da Silva Moura
* Luiz Felipe Nery Soares
* Sarah Campos Fernandes Lima
* Rafael Emanuel Dantas Viana
* Victor José Nunes Kossman

# Relatório de Implementação: Classificação Neuro-Simbólica com LTN (Logic Tensor Networks)

## 1. Introdução e Objetivo
Este experimento teve como objetivo aplicar **Logic Tensor Networks (LTN)**, uma abordagem de Inteligência Artificial Neuro-Simbólica, para resolver um problema clássico de classificação de imagens (Gatos *vs* Cachorros) utilizando o dataset **CIFAR-10**.

Diferente das redes neurais tradicionais que aprendem minimizando o erro entre um *output* e uma *label* (rótulo) numérica explícita, este modelo aprende tentando satisfazer **axiomas lógicos** definidos numa Base de Conhecimento (Knowledge Base).

## 2. Metodologia e Explicação do Código

O desenvolvimento foi dividido em cinco etapas principais:

### 2.1. Preparação dos Dados (Grounding)
O dataset CIFAR-10 foi carregado e pré-processado com as seguintes etapas:
* **Filtragem de Classes:** Foram mantidas apenas as classes "Gato" (índice 3) e "Cachorro" (índice 5).
* **Separação Lógica:** Em vez de usar rótulos `0` ou `1` para treino supervisionado padrão, os dados foram separados em dois grupos de tensores: `cats_data` e `dogs_data`.
* **Limitação:** Para fins de demonstração rápida, o dataset foi limitado a 500 amostras por classe.

### 2.2. O Predicado Neural (Modelo)
Foi definida uma Rede Neural Convolucional (CNN) simples (`SimpleCNN`), atuando como um **Predicado Unário** $Dog(x)$.
* **Entrada:** Uma imagem $x$ (3 canais, 32x32 pixels).
* **Saída:** Um valor real no intervalo $[0, 1]$ (garantido pela função de ativação *Sigmoid*).
* **Semântica:** O valor de saída representa o **grau de verdade** da afirmação "x é um cachorro".

### 2.3. Lógica Fuzzy e Conectivos
Utilizou-se a biblioteca `ltn` para definir os operadores da lógica difusa (Fuzzy Logic):
* **Not:** Negação padrão (para inverter o valor de verdade).
* **Forall:** Quantificador universal ("Para todo...").
* **SatAgg:** Agregador de satisfação, que calcula o quão bem a rede está obedecendo às regras lógicas como um todo.

### 2.4. Base de Conhecimento (Axiomas)
O aprendizado foi guiado por dois axiomas fundamentais durante o loop de treinamento.

1.  **Axioma Positivo:** $\forall x \in Dogs: Dog(x)$
    * *Significado:* Para toda imagem $x$ pertencente ao grupo de cachorros (variável `dogs_data`), o predicado deve retornar Verdadeiro (próximo de 1.0).
2.  **Axioma Negativo:** $\forall x \in Cats: \neg Dog(x)$
    * *Significado:* Para toda imagem $x$ pertencente ao grupo de gatos (variável `cats_data`), o predicado deve retornar Falso (ou seja, NÃO é um cachorro, valor próximo de 0.0).

### 2.5. Otimização
A função de perda (*Loss*) foi definida como $1 - \text{Satisfação}$. O otimizador *Adam* ajustou os pesos da CNN para maximizar a satisfação desses dois axiomas simultaneamente.

---

## 3. Análise dos Resultados

Os logs de treinamento mostram a evolução da capacidade do modelo em satisfazer as regras lógicas.

### 3.1. Evolução do Treinamento
* **Início (Época 0):**
    * `Loss: 0.4985` | `Satisfação: 0.5015`
    * **Análise:** O modelo começou com um desempenho equivalente a um "chute aleatório". A satisfação próxima de 0.5 indica que a rede não sabia distinguir as classes, atribuindo valores intermediários ou errados às imagens.
* **Meio (Época 20):**
    * `Loss: 0.2235` | `Satisfação: 0.7765`
    * **Análise:** Houve uma queda rápida na perda. O modelo começou a entender as características visuais que diferenciam as duas classes.
* **Final (Época 45):**
    * `Loss: 0.1314` | `Satisfação: 0.8686`
    * **Análise:** O modelo convergiu com uma satisfação alta (~87%). Isso significa que, na grande maioria dos casos, ele consegue afirmar corretamente que cães são cães e gatos não são cães.

### 3.2. Validação Final (Teste de Amostra)
Ao final, o modelo treinado foi submetido a um teste com imagens específicas (não vistas no cálculo do gradiente daquele passo):

| Imagem de Entrada | Predicado Testado | Valor Predito | Valor Ideal | Conclusão |
| :--- | :--- | :--- | :--- | :--- |
| **Cachorro** | $Dog(x)$ | **0.9687** | $> 0.9$ | **Correto** (Alta confiança) |
| **Gato** | $Dog(x)$ | **0.0260** | $< 0.1$ | **Correto** (Alta confiança na negação) |

O teste valida que o predicado $Dog(x)$ aprendeu corretamente a semântica desejada: ele retorna valores altos para a classe positiva e valores muito baixos para a classe negativa.

## 4. Conclusão

O experimento demonstrou com sucesso a aplicação de Redes Neurais dentro de um arcabouço lógico (LTN). O modelo não apenas classificou as imagens, mas "aprendeu a lógica" de que o conceito de Cão é oposto ao conceito de Gato dentro deste universo fechado. A convergência da satisfação para ~0.87 e os testes pontuais confirmam a eficácia da abordagem neuro-simbólica para este problema de visão computacional.
