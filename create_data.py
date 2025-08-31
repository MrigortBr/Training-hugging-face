from transformers import AutoTokenizer
import os

# Seu modelo base (o mesmo usado para o fine-tuning)
model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

# O diretório onde o seu modelo ajustado está salvo
output_dir = "./myModel/final"

# Garante que o diretório de saída existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Carrega o tokenizador do modelo base e o salva na pasta do seu modelo ajustado
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(output_dir)

print(f"Arquivos do tokenizador salvos com sucesso em {output_dir}")