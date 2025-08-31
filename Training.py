from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

def load_my_dataset():
    """Carrega o dataset personalizado de um arquivo CSV."""
    df = pd.read_csv("my_data_set.csv")
    label_mapping = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    df['label'] = df['label'].map(label_mapping)
    return Dataset.from_pandas(df)

# 1. Carregar o conjunto de dados personalizado
print("Passo 1: Carregando o conjunto de dados personalizado...")
dataset = load_my_dataset()

# Dividir o conjunto de dados em treino e validação
# Isso cria um dicionário de datasets com as chaves 'train' e 'test'
dataset_split = dataset.train_test_split(test_size=0.1)

# 2. Carregar o modelo e o tokenizador
print("Passo 2: Carregando o modelo para zero-shot...")
model_name = os.getenv("model")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 3. Pré-processar os dados
print("Passo 3: Tokenizando os dados...")
def preprocess_function(examples):
    # Combina a "hipótese" e a "premissa" em uma única sequência
    return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding="max_length")

tokenized_datasets = dataset_split.map(preprocess_function, batched=True)

# 4. Configurar o treinamento
training_args = TrainingArguments(
    output_dir="./myModel",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"], # Usamos 'test' que é a chave padrão da divisão
)

# 5. Iniciar o treinamento
print("Passo 4: Iniciando o ajuste fino...")
trainer.train()
trainer.save_model("./myModel/final")

print("Ajuste fino concluído! O modelo foi aprimorado para tarefas de NLI, o que o torna ainda melhor para zero-shot-classification.")