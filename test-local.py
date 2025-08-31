import torch
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

load_dotenv()

# Caminho para o modelo que você ajustou
model_path = "./myModel/final"

# 1. Carregar o tokenizador e o modelo ajustado
print("Passo 1: Carregando o modelo ajustado...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    print("Verifique se o caminho './myModel/final' está correto e se a pasta contém todos os arquivos (config.json, pytorch_model.bin, spiece.model).")
    exit()

# Definir os rótulos de NLI a partir da configuração do modelo
nli_labels = {id: label for id, label in model.config.id2label.items()}
entailment_id = list(nli_labels.keys())[list(nli_labels.values()).index('entailment')]
contradiction_id = list(nli_labels.keys())[list(nli_labels.values()).index('contradiction')]

# 2. Definir os textos e os rótulos candidatos para o teste
textos_para_testar = [
    "Olá, estou com uma dúvida sobre a taxa de juros do meu empréstimo pessoal.",
    "Gostaria de parabenizar a equipe de atendimento pelo excelente serviço que recebi hoje!",
    "Recebi um SMS informando sobre uma compra de alto valor que não reconheço.",
    "Tenho um problema com o meu cartão de débito, não estou conseguindo usá-lo.",
    "Feliz Natal a todos! Que o ano novo seja próspero para a empresa e para todos os colaboradores."
]

candidate_labels = ["Dúvida", "Reclamação", "Elogio", "Fraude", "Felicitacao"]

print("\n---")
print("Passo 2: Testando o modelo ajustado (Zero-Shot Simulado)...\n")
with torch.no_grad():
    for texto in textos_para_testar:
        resultados = {}
        for label in candidate_labels:
            premise = texto
            hypothesis = f"Este texto é sobre {label}."
            
            # Tokeniza e passa para o modelo
            inputs = tokenizer(premise, hypothesis, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            
            # A pontuação de "entailment" é a nossa pontuação zero-shot
            entailment_score = logits[:, entailment_id].item()
            resultados[label] = entailment_score

        # Encontrar o rótulo com a maior pontuação de implicação
        predicao_final = max(resultados, key=resultados.get)
        pontuacao_final = resultados[predicao_final]

        print(f"Texto: '{texto}'")
        print(f"Predição: {predicao_final} (Pontuação: {pontuacao_final:.4f})\n")

# 3. Testar o modelo original do Hugging Face (para comparação)
print("---")
print("Passo 3: Testando o modelo original do Hub para comparação...\n")

model_name_original = os.getenv("model")
try:
    classifier_original = pipeline("zero-shot-classification", model=model_name_original)
    resultados_originais = classifier_original(textos_para_testar, candidate_labels=candidate_labels)
    for texto, resultado in zip(textos_para_testar, resultados_originais):
        print(f"Texto: '{texto}'")
        print(f"Predição: {resultado['labels'][0]} (Pontuação: {resultado['scores'][0]:.4f})\n")
except Exception as e:
    print(f"Não foi possível carregar ou testar o modelo original. Erro: {e}")