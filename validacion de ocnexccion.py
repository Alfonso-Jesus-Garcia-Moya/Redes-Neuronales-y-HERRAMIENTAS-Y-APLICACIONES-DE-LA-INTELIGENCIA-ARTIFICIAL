import requests
import logging
from typing import Dict, Any

# Configuración de telemetría estándar para consola
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

API_KEY = "3zowvptAToWZwewH83aidKGAS24bW66zsE5dhgw7PEUpJkX9LQodJQQJ99CBACHYHv6XJ3w3AAAAAC0GqkBQ"

def validar_azure_anthropic(deployment_name: str) -> None:
    """
    Valida el endpoint de API Management configurado para enrutar tráfico a modelos Anthropic.
    Fuerza los headers tanto de Azure como de Anthropic para sortear bloqueos de WAF.
    """
    url = "https://alfonsinnova2026.openai.azure.com/anthropic/v1/messages"
    
    headers: Dict[str, str] = {
        "api-key": API_KEY,           # Autenticación nativa de Azure
        "x-api-key": API_KEY,         # Contingencia si el wrapper espera formato Anthropic
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    
    payload: Dict[str, Any] = {
        "model": deployment_name,
        "max_tokens": 15,
        "messages": [{"role": "user", "content": "Prueba de conectividad HTTP. Responde 'OK'."}]
    }
    
    logging.info(f"Iniciando probe hacia Anthropic Gateway -> Modelo: {deployment_name}")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            respuesta_texto = response.json().get('content', [{}])[0].get('text', 'Sin texto')
            logging.info(f"ÉXITO (200): Conexión establecida. Respuesta del modelo: {respuesta_texto}")
        elif response.status_code == 401:
            logging.error("FALLO (401): Credenciales rechazadas. Verifica si la API Key fue rotada o está inactiva.")
        elif response.status_code == 404:
            logging.error(f"FALLO (404): Endpoint no encontrado. Verifica que el deployment_name '{deployment_name}' exista en tu recurso de Azure.")
        else:
            logging.error(f"FALLO ({response.status_code}): {response.text}")
            
    except requests.exceptions.Timeout:
        logging.critical("TIMEOUT: El servidor no respondió en 10 segundos. Posible bloqueo de firewall o proxy.")
    except requests.exceptions.ConnectionError:
        logging.critical("CONNECTION ERROR: Fallo en la resolución DNS o conexión rechazada a nivel de socket.")
    except Exception as e:
        logging.critical(f"ERROR FATAL: {str(e)}")

def validar_azure_openai(deployment_name: str, api_version: str = "2025-04-01-preview") -> None:
    """
    Valida el endpoint nativo de Azure Cognitive Services estructurando la URL 
    correctamente bajo el esquema /deployments/{model}/chat/completions
    """
    url = f"https://alfonsinnova2026.cognitiveservices.azure.com/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"
    
    headers: Dict[str, str] = {
        "api-key": API_KEY,
        "Content-Type": "application/json"
    }
    
    payload: Dict[str, Any] = {
        "messages": [{"role": "user", "content": "Prueba de conectividad HTTP. Responde 'OK'."}],
        "max_tokens": 15
    }
    
    logging.info(f"Iniciando probe hacia Azure OpenAI Nativo -> Modelo: {deployment_name}")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            respuesta_texto = response.json().get('choices', [{}])[0].get('message', {}).get('content', 'Sin texto')
            logging.info(f"ÉXITO (200): Conexión establecida. Respuesta del modelo: {respuesta_texto}")
        else:
            logging.error(f"FALLO ({response.status_code}): {response.text}")
            
    except Exception as e:
         logging.critical(f"ERROR DE RED CRÍTICO: {str(e)}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print(" EJECUCIÓN DE DIAGNÓSTICO DE INFRAESTRUCTURA DE INFERENCIA")
    print("="*60 + "\n")
    
    # 1. Pruebas para modelos Claude (Endpoint /anthropic)
    validar_azure_anthropic("claude-sonnet-4-6")
    validar_azure_anthropic("claude-opus-4-7-2")
    
    print("\n" + "-"*60 + "\n")
    
    # 2. Pruebas para modelos GPT (Endpoint nativo Azure)
    validar_azure_openai("gpt-5.2-codex")
    validar_azure_openai("gpt-5.1-codex-max")
    
    print("\n" + "="*60 + "\n")