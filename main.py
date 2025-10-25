# main.py: Simulación en terminal (para testing inicial)
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from agent import build_graph
from rag import setup_chroma

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Configura GEMINI_API_KEY en .env")

# Asume tu documento FAQ (cambia el path)
document_path = "faqs.txt"  # O 'faqs.txt'; cárgalo aquí
retriever = setup_chroma(document_path)

graph = build_graph(retriever, api_key)

def simulate_conversation():
    """Simula input/output en terminal, como Twilio."""
    print("Bienvenido al agente FAQ. Escribe 'exit' para salir.")
    state = {"messages": []}
    
    while True:
        user_input = input("Tú: ")
        if user_input.lower() == 'exit':
            break
        
        state["messages"].append(HumanMessage(content=user_input))
        
        for updated_state in graph.stream(state, stream_mode="values"):
            last_message = updated_state["messages"][-1]
            print(f"Agente: {last_message.content}")
        
        state = updated_state  # Mantiene estado para conversación

if __name__ == "__main__":
    simulate_conversation()