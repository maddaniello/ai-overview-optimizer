"""
Base Agent - Classe base per tutti gli agenti specializzati
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum
import asyncio


class AgentStatus(Enum):
    """Stato dell'agente"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    WAITING = "waiting"


@dataclass
class AgentMessage:
    """Messaggio scambiato tra agenti"""
    sender: str
    recipient: str
    content: Any
    message_type: str = "info"  # info, task, result, error, log
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "message_type": self.message_type,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class AgentState:
    """Stato condiviso tra agenti"""
    # Input
    keyword: str = ""
    target_url: Optional[str] = None
    user_answer: Optional[str] = None
    location: str = "Italy"
    language: str = "Italian"

    # Config
    model_id: str = "gpt-4o"
    max_iterations: int = 3
    max_serp_results: int = 10
    max_sources: int = 5

    # API Keys
    openai_api_key: str = ""
    gemini_api_key: str = ""
    dataforseo_login: str = ""
    dataforseo_password: str = ""

    # Google Cloud Ranking (optional)
    google_project_id: str = ""
    google_credentials_json: str = ""
    ranking_method: str = "embeddings"  # "embeddings" or "google"

    # SERP Data
    serp_data: Optional[Dict] = None
    ai_overview_text: str = ""
    ai_overview_sources: List[Dict] = field(default_factory=list)
    organic_results: List[Dict] = field(default_factory=list)
    fan_out_queries: List[str] = field(default_factory=list)

    # Scraped Content
    target_content: str = ""
    competitor_contents: List[Dict] = field(default_factory=list)

    # Ranking & Analysis
    initial_ranking: List[Dict] = field(default_factory=list)
    current_ranking: List[Dict] = field(default_factory=list)
    entity_gap: Dict = field(default_factory=dict)

    # Optimization
    iterations: List[Dict] = field(default_factory=list)
    current_answer: str = ""
    best_answer: str = ""
    best_score: float = 0.0

    # Strategy
    strategic_analysis: Optional[Dict] = None
    content_plan: Optional[Dict] = None

    # Logs
    logs: List[Dict] = field(default_factory=list)

    def add_log(self, agent: str, message: str, level: str = "info", data: Any = None):
        """Aggiunge un log"""
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "level": level,
            "message": message,
            "data": data
        })

    def get_provider(self) -> str:
        """Ritorna il provider del modello selezionato"""
        from config import ALL_MODELS
        model_config = ALL_MODELS.get(self.model_id, {})
        return model_config.get("provider", "openai")


class BaseAgent(ABC):
    """
    Classe base astratta per tutti gli agenti.
    Ogni agente specializzato deve implementare il metodo execute().
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        log_callback: Optional[Callable] = None
    ):
        """
        Args:
            name: Nome univoco dell'agente
            description: Descrizione del ruolo dell'agente
            log_callback: Callback per inviare log in tempo reale
        """
        self.name = name
        self.description = description
        self.status = AgentStatus.IDLE
        self.log_callback = log_callback
        self.messages: List[AgentMessage] = []

    def log(self, message: str, level: str = "info", data: Any = None):
        """Invia un log"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": self.name,
            "level": level,
            "message": message,
            "data": data
        }

        if self.log_callback:
            self.log_callback(log_entry)

        return log_entry

    def send_message(self, recipient: str, content: Any, message_type: str = "info", metadata: Dict = None) -> AgentMessage:
        """Invia un messaggio ad un altro agente"""
        msg = AgentMessage(
            sender=self.name,
            recipient=recipient,
            content=content,
            message_type=message_type,
            metadata=metadata or {}
        )
        self.messages.append(msg)
        return msg

    @abstractmethod
    async def execute(self, state: AgentState) -> AgentState:
        """
        Esegue il task dell'agente.
        Deve essere implementato da ogni agente specializzato.

        Args:
            state: Stato condiviso corrente

        Returns:
            Stato aggiornato
        """
        pass

    async def run(self, state: AgentState) -> AgentState:
        """
        Wrapper per execute() con gestione errori e logging.
        """
        self.status = AgentStatus.RUNNING
        self.log(f"Agente {self.name} avviato", level="info")

        try:
            state = await self.execute(state)
            self.status = AgentStatus.COMPLETED
            self.log(f"Agente {self.name} completato", level="success")

        except Exception as e:
            self.status = AgentStatus.ERROR
            self.log(f"Errore in {self.name}: {str(e)}", level="error", data={"error": str(e)})
            state.add_log(self.name, f"Errore: {str(e)}", level="error")
            raise

        return state

    def __repr__(self):
        return f"<{self.__class__.__name__}(name='{self.name}', status={self.status.value})>"
