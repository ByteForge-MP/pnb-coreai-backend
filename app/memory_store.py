
conversation_memory = []

def add_memory(user, assistant):
    conversation_memory.append({
        "user": user,
        "assistant": assistant
    })

def get_memory():
    history = ""
    for m in conversation_memory[-3:]:
        history += f"""
                       User: {m['user']}
                       Assistant: {m['assistant']}
                    """
    return history