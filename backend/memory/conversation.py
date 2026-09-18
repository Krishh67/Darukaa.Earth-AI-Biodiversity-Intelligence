import uuid

class ConversationMemory:
    def __init__(self):
        self.sessions = {}
        
    def get_or_create(self, session_id=None):
        if not session_id or session_id not in self.sessions:
            session_id = str(uuid.uuid4())
            self.sessions[session_id] = []
        return session_id
        
    def add_message(self, session_id, role, content):
        if session_id in self.sessions:
            self.sessions[session_id].append({"role": role, "content": content})
            # Keep only the last 10 turns to save context length
            if len(self.sessions[session_id]) > 20:
                self.sessions[session_id] = self.sessions[session_id][-20:]
                
    def get_history(self, session_id):
        return self.sessions.get(session_id, [])

memory = ConversationMemory()

