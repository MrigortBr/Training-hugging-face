from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Carregar o modelo e o tokenizador do seu diretório local
model_name = "./myModel/final"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Fazer o upload para o Hub
repo_name = "igortbr/nli-finetuned-email-model"

model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)

