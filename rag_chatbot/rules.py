# rag_chatbot/rules.py

RULES = {
    "who are you": "I am the Academic RAG Assistant for President University.",
    "jam berapa": "Kami buka dari jam 09:00 - 16:00.",
    "contact": "You can contact the Academic Bureau at academic@president.ac.id."
}

def check_rules(question: str):
    """
    Returns an answer if a rule matches, otherwise returns None.
    """
    question_lower = question.lower().strip()
    
    # Simple keyword matching
    for key, answer in RULES.items():
        if key in question_lower:
            return answer
    
    return None