import os
import json
import uuid
from datetime import datetime

class ConversationMemory:
    def __init__(self, data_dir="data/sessions"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
    def get_or_create(self, session_id=None):
        if not session_id or not self._exists(session_id):
            session_id = str(uuid.uuid4())
            self._save(session_id, {"created_at": datetime.now().isoformat(), "history": []})
        return session_id
        
    def _exists(self, session_id):
        return os.path.exists(os.path.join(self.data_dir, f"{session_id}.json"))
        
    def _save(self, session_id, data):
        with open(os.path.join(self.data_dir, f"{session_id}.json"), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
    def _load(self, session_id):
        if self._exists(session_id):
            try:
                with open(os.path.join(self.data_dir, f"{session_id}.json"), 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {"created_at": datetime.now().isoformat(), "history": [], "initialized": False}
        
    def is_initialized(self, session_id):
        return self._load(session_id).get("initialized", False)
        
    def set_initialized(self, session_id):
        data = self._load(session_id)
        data["initialized"] = True
        self._save(session_id, data)
        
    def add_message(self, session_id, role, content):
        data = self._load(session_id)
        data["history"].append({"role": role, "content": content})
        # Keep last 20 messages for context window size limits
        if len(data["history"]) > 20:
            data["history"] = data["history"][-20:]
        data["updated_at"] = datetime.now().isoformat()
        self._save(session_id, data)
        
    def get_history(self, session_id):
        data = self._load(session_id)
        return data.get("history", [])

    def list_sessions(self):
        sessions = []
        for f in os.listdir(self.data_dir):
            if f.endswith('.json'):
                sid = f.replace('.json', '')
                data = self._load(sid)
                
                # Title is based on first user message
                hist = data.get("history", [])
                title = "New Conversation"
                for m in hist:
                    if m["role"] == "user":
                        title = m["content"][:40] + "..." if len(m["content"]) > 40 else m["content"]
                        break
                        
                updated = data.get("updated_at", data.get("created_at", ""))
                sessions.append({"id": sid, "title": title, "updated_at": updated})
                
        # Sort desc by updated_at
        sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return sessions

memory = ConversationMemory()
