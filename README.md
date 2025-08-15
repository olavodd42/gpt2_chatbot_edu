# Chatbot Educacional com GPT-2 + LoRA + 8-bit

Fine-tuning do GPT‑2 (com LoRA + 8‑bit via bitsandbytes) para diálogo educacional, usando transformers, datasets, peft e accelerate.

- **LoRA** (Low-Rank Adaptation) para treinamento eficiente
- **Quantização em 8 bits** via **BitsAndBytes**
- **Hugging Face Transformers/Datasets**
- **W&B** (Weights & Biases) para logging de métricas e amostras
- Suporte a datasets de **perguntas e respostas** no formato de **diálogo** (`<|user|>` e `<|assistant|>`)

O objetivo é treinar um modelo rápido, leve e personalizável para uso em ambientes educacionais e pesquisa.

Licença: ver [LICENSE.md](LICENSE.md)

---

## Sumário

- [📂 Estrutura do Projeto](#-estrutura-do-projeto)
- [⚙️ Funcionalidades](#️-funcionalidades)
- [📦 Instalação](#-instalação)
- [🛠️ Requisitos](#️-requisitos)
- [📥 Preparação de Dados](#-preparação-de-dados)
- [🏋️ Treinamento](#-treinamento)
- [🤖 Inferência](#-inferência)
- [📊 Estrutura de Configuração](#-estrutura-de-configuração-train_smallyaml)
- [🧪 Estrutura de Tokenização](#-estrutura-de-tokenização)
- [📈 Integração com Weights & Biases](#-integração-com-weights--biases)
- [📜 Exemplo de Dataset no Formato Final](#-exemplo-de-dataset-no-formato-final)
- [📚 Possíveis Extensões para Artigo](#-possíveis-extensões-para-artigo)
- [📊 Metodologia e Resultados](#-metodologia-e-resultados)
- [📄 Citar este trabalho](#-citar-este-trabalho)
- [📜 Licença](#-licença)

---

## 📂 Estrutura do Projeto
```
gpt2_chatbot_edu/
│
├── configs/
│   └── train_small.yaml
│
├── scripts/
│   ├── 01_prepare_datasets.py
│   ├── 02_train.py
│   └── 03_inference.py
│
├── src/
│   ├── data_prep.py
│   ├── modeling.py
│   ├── train_loop.py
│   ├── chat_loop.py
│   ├── templates.py
│   └── utils.py
│
├── LICENSE.md
├── README.md
└── requirements.txt
```

---

## ⚙️ Funcionalidades

- **Treinamento eficiente** com LoRA + quantização 8-bit
- **Formato de diálogo padronizado**:
  ```txt
  <|user|> Pergunta
  <|assistant|> Resposta
  ```
- **Limpeza de texto**: remoção de HTML, caracteres estranhos e normalização de espaços
- **Conversão de datasets externos** (Natural Questions, SciQ, Dolly, Alpaca)
- **Logging no W&B** de métricas, amostras e hiperparâmetros
- **Inferência interativa** com stopping criteria customizado

## 📦 Instalação
```bash
git clone https://github.com/SEU-USUARIO/gpt2_chatbot_edu.git
cd gpt2_chatbot_edu
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 🛠️ Requisitos
- Python 3.10+
- torch >=2.0.0
- transformers >=4.40.0
- datasets >=2.19.0
- peft >=0.10.0
- bitsandbytes >=0.43.0
- accelerate >=0.30.0
- wandb >=0.17.0
- beautifulsoup4 >=4.12.0
- PyYAML >=6.0

## 📊 Estrutura de Configuração (train_small.yaml)
*(Veja exemplo completo no repositório)*

```yaml
project_name: chatbot-edu-gpt2
model:
  base_ckpt: gpt2
lora:
  enabled: True
train:
  epochs: 3
  lr: 5e-5
logging:
  report_to: "wandb"
```

## 📥 Preparação de Dados
Script principal: [`scripts/01_prepare_datasets.py`](scripts/01_prepare_datasets.py)
```bash
python scripts/01_prepare_datasets.py
```
Gera arquivos `train.jsonl`, `val.jsonl` e `test.jsonl` no diretório atual e exibe a distribuição de amostras por fonte.

## 🏋️ Treinamento
Script principal: [`scripts/02_train.py`](scripts/02_train.py)
```bash
python scripts/02_train.py
```
Carrega as configurações de `configs/train_small.yaml`, tokeniza os dados, avalia o modelo base e realiza o fine-tuning com LoRA+8-bit. Ao final, imprime métricas como `eval_loss` e `ppl` e salva pesos e tokenizer em `experiments/`.

## 🤖 Inferência
Script principal: [`scripts/03_inference.py`](scripts/03_inference.py)
```bash
python scripts/03_inference.py
```
Compara respostas do modelo base e do modelo ajustado para uma pergunta de exemplo e gera saídas para o conjunto de teste.

## 🧪 Estrutura de Tokenização
```python
tokenized_dataset = tokenize_dataset(tokenizer, dataset)
print(tokenized_dataset["train"].column_names)  # ['input_ids', 'attention_mask']
```

## 📈 Integração com Weights & Biases
```yaml
logging:
  report_to: "wandb"
```
```python
from src.train_loop import log_samples_wandb
log_samples_wandb(trainer, tokenizer, prompts)
```

## 📜 Exemplo de Dataset no Formato Final
```json
{"dialogue": [
  {"role": "user", "content": "What is the capital of France?"},
  {"role": "assistant", "content": "Paris."}
]}
```
Formato texto:
```
<|user|> What is the capital of France?
<|assistant|> Paris.
```

## 📚 Possíveis Extensões para Artigo
- Comparar LoRA vs Full Fine-tuning
- Quantização 8-bit vs 16-bit
- Avaliar em diferentes domínios
- Analisar custo computacional vs qualidade

## 📊 Metodologia e Resultados
**Metodologia:**
1. Coleta e conversão de datasets QA para formato de diálogo
2. Fine-tuning do GPT-2 com LoRA + 8-bit
3. Uso de W&B para logging e análise
4. Implementação de StoppingCriteria para evitar vazamentos

**Resultados:**
- **Perda (eval_loss)** e **Perplexidade (PPL)** melhoraram em relação ao modelo base
- Respostas mais concisas e contextualizadas
- Redução de custo de GPU com LoRA + 8-bit

## 📄 Citar este trabalho
```
Olavo Dalberto (2025). gpt2_chatbot_edu: Fine-tuning GPT-2 with LoRA+8-bit for dialog.
https://github.com/olavodd42/gpt2_chatbot_edu
```

## 📜 Licença
Este projeto está licenciado sob a MIT License. Veja [LICENSE.md](LICENSE.md) para mais detalhes.