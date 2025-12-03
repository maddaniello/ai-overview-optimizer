"""
LLM Client - Client unificato per OpenAI e Google Gemini
"""
from typing import Dict, Any, Optional, List
from openai import OpenAI
from utils.logger import logger
from config import ALL_MODELS, OPENAI_MODELS, GEMINI_MODELS


class LLMClient:
    """
    Client unificato per interagire con diversi provider LLM.
    Supporta: OpenAI (GPT-4, o1) e Google Gemini
    """

    def __init__(
        self,
        model_id: str = "gpt-4o",
        openai_api_key: str = None,
        gemini_api_key: str = None
    ):
        """
        Args:
            model_id: ID del modello da usare
            openai_api_key: API key OpenAI
            gemini_api_key: API key Google Gemini
        """
        self.model_id = model_id
        self.openai_api_key = openai_api_key
        self.gemini_api_key = gemini_api_key

        # Determina provider
        model_config = ALL_MODELS.get(model_id, {})
        self.provider = model_config.get("provider", "openai")
        self.max_tokens = model_config.get("max_tokens", 4096)

        # Inizializza client
        self.openai_client = None
        self.gemini_model = None

        if self.provider == "openai" and openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
            logger.info(f"LLM Client inizializzato: {model_id} (OpenAI)")

        elif self.provider == "gemini" and gemini_api_key:
            self._init_gemini(gemini_api_key)
            logger.info(f"LLM Client inizializzato: {model_id} (Gemini)")

        else:
            # Fallback a OpenAI se disponibile
            if openai_api_key:
                self.provider = "openai"
                self.model_id = "gpt-4o"
                self.openai_client = OpenAI(api_key=openai_api_key)
                logger.warning(f"Fallback a OpenAI {self.model_id}")

    def _init_gemini(self, api_key: str):
        """Inizializza Google Gemini"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel(self.model_id)
            logger.info(f"Gemini model caricato: {self.model_id}")
        except ImportError:
            logger.warning("google-generativeai non installato - fallback a OpenAI")
            self.provider = "openai"
        except Exception as e:
            logger.error(f"Errore init Gemini: {e}")
            self.provider = "openai"

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = None,
        system_prompt: str = None
    ) -> str:
        """
        Invia una chat request al modello.

        Args:
            messages: Lista di messaggi [{"role": "user", "content": "..."}]
            temperature: Temperatura (creatività)
            max_tokens: Limite token risposta
            system_prompt: System prompt opzionale

        Returns:
            Risposta del modello
        """
        max_tokens = max_tokens or self.max_tokens

        if self.provider == "openai":
            return self._chat_openai(messages, temperature, max_tokens, system_prompt)
        elif self.provider == "gemini":
            return self._chat_gemini(messages, temperature, max_tokens, system_prompt)
        else:
            raise ValueError(f"Provider non supportato: {self.provider}")

    def _chat_openai(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        system_prompt: str = None
    ) -> str:
        """Chat con OpenAI"""
        if not self.openai_client:
            raise ValueError("OpenAI client non inizializzato")

        # Prepara messaggi
        full_messages = []

        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})

        full_messages.extend(messages)

        # o1 models non supportano system prompt e temperature
        is_o1_model = self.model_id.startswith("o1")

        try:
            if is_o1_model:
                # o1 models: no system, no temperature
                response = self.openai_client.chat.completions.create(
                    model=self.model_id,
                    messages=full_messages,
                    max_completion_tokens=max_tokens
                )
            else:
                response = self.openai_client.chat.completions.create(
                    model=self.model_id,
                    messages=full_messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Errore OpenAI chat: {e}")
            raise

    def _chat_gemini(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        system_prompt: str = None
    ) -> str:
        """Chat con Google Gemini"""
        if not self.gemini_model:
            raise ValueError("Gemini model non inizializzato")

        try:
            import google.generativeai as genai

            # Converti messaggi in formato Gemini
            # Gemini usa "parts" invece di "content"
            gemini_messages = []

            if system_prompt:
                # Gemini gestisce system prompt come primo messaggio
                gemini_messages.append({
                    "role": "user",
                    "parts": [f"[System Instructions]\n{system_prompt}"]
                })
                gemini_messages.append({
                    "role": "model",
                    "parts": ["Capito, seguirò queste istruzioni."]
                })

            for msg in messages:
                role = "user" if msg["role"] == "user" else "model"
                gemini_messages.append({
                    "role": role,
                    "parts": [msg["content"]]
                })

            # Genera risposta
            chat = self.gemini_model.start_chat(history=gemini_messages[:-1])
            response = chat.send_message(
                gemini_messages[-1]["parts"][0],
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )

            return response.text.strip()

        except Exception as e:
            logger.error(f"Errore Gemini chat: {e}")
            raise

    def generate_with_reasoning(
        self,
        prompt: str,
        context: str = "",
        task_description: str = "",
        temperature: float = 0.7
    ) -> Dict[str, str]:
        """
        Genera risposta con ragionamento esplicito (chain-of-thought).

        Returns:
            Dict con "reasoning" e "answer"
        """
        system_prompt = """Sei un esperto SEO content strategist.
Quando rispondi, segui SEMPRE questo formato:

## RAGIONAMENTO
[Spiega passo dopo passo il tuo ragionamento]

## RISPOSTA
[La tua risposta finale]

Sii dettagliato nel ragionamento per mostrare come arrivi alle conclusioni."""

        full_prompt = f"""
{f"CONTESTO: {context}" if context else ""}
{f"TASK: {task_description}" if task_description else ""}

{prompt}
"""

        response = self.chat(
            messages=[{"role": "user", "content": full_prompt}],
            system_prompt=system_prompt,
            temperature=temperature
        )

        # Estrai ragionamento e risposta
        reasoning = ""
        answer = response

        if "## RAGIONAMENTO" in response and "## RISPOSTA" in response:
            parts = response.split("## RISPOSTA")
            reasoning = parts[0].replace("## RAGIONAMENTO", "").strip()
            answer = parts[1].strip() if len(parts) > 1 else response

        return {
            "reasoning": reasoning,
            "answer": answer,
            "full_response": response
        }

    def get_embedding(self, text: str) -> List[float]:
        """Genera embedding per un testo (solo OpenAI)"""
        if not self.openai_client:
            raise ValueError("OpenAI client necessario per embeddings")

        from config import OPENAI_EMBEDDING_MODEL

        response = self.openai_client.embeddings.create(
            input=text,
            model=OPENAI_EMBEDDING_MODEL
        )

        return response.data[0].embedding

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Genera embeddings per multipli testi"""
        if not self.openai_client:
            raise ValueError("OpenAI client necessario per embeddings")

        from config import OPENAI_EMBEDDING_MODEL

        response = self.openai_client.embeddings.create(
            input=texts,
            model=OPENAI_EMBEDDING_MODEL
        )

        return [data.embedding for data in response.data]
