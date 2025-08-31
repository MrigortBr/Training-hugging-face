from dotenv import load_dotenv
from transformers import pipeline
import os

load_dotenv()

model_name_original = "igortbr/nli-finetuned-email-model"

textos_para_testar = [
    "Olá, estou com uma dúvida sobre a taxa de juros do meu empréstimo pessoal.",
    "Gostaria de parabenizar a equipe de atendimento pelo excelente serviço que recebi hoje!",
    "Recebi um SMS informando sobre uma compra de alto valor que não reconheço.",
    "Tenho um problema com o meu cartão de débito, não estou conseguindo usá-lo.",
    "Feliz Natal a todos! Que o ano novo seja próspero para a empresa e para todos os colaboradores."
]

candidate_labels = [
  "Solicitação de serviço",
  "Problema técnico",
  "Pedido de informação",
  "Problema financeiro",
  "Pedido de documento",
  "Pedido de cancelamento",
  "Problema de acesso",
  "Alerta de segurança",
  "Agradecimento",
  "Elogio",
  "Congratulação",
  "Felicitações de feriado"
]

print(f"Modelo {model_name_original}\n")

try:
    classifier_original = pipeline("zero-shot-classification", model=model_name_original)
    resultados_originais = classifier_original(textos_para_testar, candidate_labels=candidate_labels)
    for texto, resultado in zip(textos_para_testar, resultados_originais):
        print(f"Texto: '{texto}'")
        print(f"Predição: {resultado['labels'][0]} (Pontuação: {resultado['scores'][0]:.4f})\n")
except Exception as e:
    print(f"Não foi possível carregar ou testar o modelo original. Erro: {e}")

print(f"Modelo {os.getenv('model')}\n")


