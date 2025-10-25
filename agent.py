# agent.py: Configuración de LangGraph con RAG (Personalidad Banorte)
from typing import Annotated, Sequence, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableConfig
import os
from dotenv import load_dotenv
from rag import setup_chroma

load_dotenv()

# Estado del agente (con mensajes para conversación)
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def build_graph(retriever, api_key: str):
    """Construye el gráfico de LangGraph con RAG."""
    # Inicializa LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        google_api_key=api_key,
    )
    
    def rag_node(state: AgentState, config: RunnableConfig):
        """Nodo RAG: Recupera contexto y genera respuesta como agente de Banorte."""
        try:
            question = state["messages"][-1].content
            # Construye el historial de la conversación
            history = "\n".join(
                f"{'Usuario' if msg.type == 'human' else 'Asistente de Banorte'}: {msg.content}"
                for msg in state["messages"][:-1]  # Excluye la última pregunta
            ) if state["messages"][:-1] else "No hay historial previo."
            
            # Prompt específico para preguntas sobre el historial
            if any(keyword in question.lower() for keyword in ["qué te había preguntado", "pregunta anterior", "de qué hablamos"]):
                prompt = ChatPromptTemplate.from_template(
                    "Eres un agente de soporte virtual de Banorte.\n"
                    "Historial de la conversación:\n{history}\n\n"
                    "Pregunta actual: {question}\n\n"
                    "Responde en español, usando SOLO el historial de la conversación para identificar la pregunta o tema anterior. "
                    "Si no hay historial, di que es la primera pregunta que recibes. "
                    "NO inventes información."
                )
                chain = prompt | llm
                response = chain.invoke({"history": history, "question": question})
            else:
                # Prompt para preguntas generales con contexto RAG (Persona Banorte)
                relevant_docs = retriever.invoke(question)
                context = "\n\n".join(doc.page_content for doc in relevant_docs)
                
                prompt_template = (
                    "Eres un agente de soporte virtual de Banorte. Tu propósito es ayudar a los usuarios con preguntas sobre la aplicación y los servicios de Banorte.\n\n"
                    "Historial de la conversación:\n{history}\n\n"
                    "Contexto de FAQs (Preguntas Frecuentes):\n{context}\n\n"
                    "Pregunta actual: {question}\n\n"
                    "Instrucciones:\n"
                    "1. Responde en español de manera amable y profesional, como un agente de Banorte.\n"
                    "2. Usa el 'Contexto de FAQs' como tu fuente principal de verdad.\n"
                    "3. Usa el 'Historial de la conversación' para entender preguntas de seguimiento.\n"
                    "4. Si la respuesta NO está en el contexto, indica amablemente que no tienes esa información y que el usuario puede consultar el sitio web oficial de Banorte o a un asesor humano.\n"
                    "5. NO inventes información que no esté en el contexto."
                )
                
                prompt = ChatPromptTemplate.from_template(prompt_template)
                chain = prompt | llm
                response = chain.invoke({"context": context, "history": history, "question": question})
            
            return {"messages": [AIMessage(content=response.content)]}
        except Exception as e:
            return {"messages": [AIMessage(content=f"Error en RAG: {str(e)}")]}

    def should_continue(state: AgentState):
        """Decide si terminar (por ahora siempre, ya que es simple RAG)."""
        return END

    # Construye graph
    workflow = StateGraph(AgentState)
    workflow.add_node("rag", rag_node)
    workflow.set_entry_point("rag")
    workflow.add_conditional_edges("rag", should_continue)
    
    return workflow.compile()