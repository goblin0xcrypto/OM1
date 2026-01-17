"""
Conversation History Persistence Module for OM1
Saves conversation history to disk and reloads on startup/crash recovery
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import threading
import logging

logger = logging.getLogger(__name__)


class ConversationHistoryManager:
    """Manages conversation history persistence with automatic save/load functionality"""
    
    def __init__(
        self, 
        history_dir: str = "conversation_history",
        agent_name: str = "default_agent",
        auto_save: bool = True,
        max_history_size: int = 100
    ):
        """
        Initialize the conversation history manager
        
        Args:
            history_dir: Directory to store conversation history files
            agent_name: Name of the agent (used for file naming)
            auto_save: Whether to automatically save after each message
            max_history_size: Maximum number of messages to keep in history
        """
        self.history_dir = Path(history_dir)
        self.agent_name = agent_name
        self.auto_save = auto_save
        self.max_history_size = max_history_size
        
        # Create history directory if it doesn't exist
        self.history_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize conversation history
        self.history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        
        # Load existing history if available
        self._load_history()
        
        logger.info(f"ConversationHistoryManager initialized for agent '{agent_name}'")
        logger.info(f"Loaded {len(self.history)} messages from previous session")
    
    @property
    def history_file(self) -> Path:
        """Get the path to the history file for this agent"""
        return self.history_dir / f"{self.agent_name}_history.json"
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """
        Add a message to the conversation history
        
        Args:
            role: The role of the speaker (e.g., 'user', 'assistant', 'system')
            content: The message content
            metadata: Optional metadata about the message
        """
        with self._lock:
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            self.history.append(message)
            
            # Trim history if it exceeds max size
            if len(self.history) > self.max_history_size:
                self.history = self.history[-self.max_history_size:]
                logger.debug(f"Trimmed history to {self.max_history_size} messages")
            
            # Auto-save if enabled
            if self.auto_save:
                self._save_history()
    
    def get_history(
        self, 
        limit: Optional[int] = None,
        role_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history
        
        Args:
            limit: Maximum number of recent messages to return
            role_filter: Only return messages from this role
            
        Returns:
            List of message dictionaries
        """
        with self._lock:
            history = self.history.copy()
            
            # Filter by role if specified
            if role_filter:
                history = [msg for msg in history if msg["role"] == role_filter]
            
            # Limit to most recent messages if specified
            if limit:
                history = history[-limit:]
            
            return history
    
    def get_formatted_history(
        self, 
        limit: Optional[int] = None,
        include_system: bool = True
    ) -> List[Dict[str, str]]:
        """
        Get conversation history formatted for LLM input
        
        Args:
            limit: Maximum number of recent messages to return
            include_system: Whether to include system messages
            
        Returns:
            List of messages in LLM-friendly format
        """
        with self._lock:
            history = self.history.copy()
            
            # Filter out system messages if not needed
            if not include_system:
                history = [msg for msg in history if msg["role"] != "system"]
            
            # Limit to most recent messages if specified
            if limit:
                history = history[-limit:]
            
            # Format for LLM
            formatted = [
                {
                    "role": msg["role"],
                    "content": msg["content"]
                }
                for msg in history
            ]
            
            return formatted
    
    def clear_history(self, save_backup: bool = True) -> None:
        """
        Clear the conversation history
        
        Args:
            save_backup: Whether to save a backup before clearing
        """
        with self._lock:
            if save_backup and self.history:
                backup_file = self.history_dir / f"{self.agent_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self._save_to_file(backup_file)
                logger.info(f"Saved backup to {backup_file}")
            
            self.history = []
            self._save_history()
            logger.info("Conversation history cleared")
    
    def _save_history(self) -> None:
        """Save conversation history to disk"""
        try:
            self._save_to_file(self.history_file)
            logger.debug(f"Saved {len(self.history)} messages to {self.history_file}")
        except Exception as e:
            logger.error(f"Failed to save conversation history: {e}")
    
    def _save_to_file(self, filepath: Path) -> None:
        """Save conversation history to a specific file"""
        data = {
            "agent_name": self.agent_name,
            "saved_at": datetime.now().isoformat(),
            "message_count": len(self.history),
            "history": self.history
        }
        
        # Write to temporary file first, then rename (atomic operation)
        temp_file = filepath.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Atomic rename
        temp_file.replace(filepath)
    
    def _load_history(self) -> None:
        """Load conversation history from disk"""
        if not self.history_file.exists():
            logger.info("No existing conversation history found")
            return
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.history = data.get("history", [])
            
            # Validate and clean history
            self.history = [
                msg for msg in self.history
                if isinstance(msg, dict) and "role" in msg and "content" in msg
            ]
            
            logger.info(f"Loaded {len(self.history)} messages from {self.history_file}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse conversation history: {e}")
            logger.info("Starting with empty history")
        except Exception as e:
            logger.error(f"Failed to load conversation history: {e}")
            logger.info("Starting with empty history")
    
    def export_history(self, filepath: str) -> None:
        """
        Export conversation history to a file
        
        Args:
            filepath: Path to export file
        """
        with self._lock:
            export_path = Path(filepath)
            self._save_to_file(export_path)
            logger.info(f"Exported conversation history to {export_path}")
    
    def import_history(self, filepath: str, append: bool = False) -> None:
        """
        Import conversation history from a file
        
        Args:
            filepath: Path to import file
            append: Whether to append to existing history or replace it
        """
        import_path = Path(filepath)
        if not import_path.exists():
            logger.error(f"Import file not found: {import_path}")
            return
        
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            imported_history = data.get("history", [])
            
            with self._lock:
                if append:
                    self.history.extend(imported_history)
                else:
                    self.history = imported_history
                
                # Trim if needed
                if len(self.history) > self.max_history_size:
                    self.history = self.history[-self.max_history_size:]
                
                self._save_history()
                logger.info(f"Imported {len(imported_history)} messages from {import_path}")
                
        except Exception as e:
            logger.error(f"Failed to import conversation history: {e}")


