# api.py: FastAPI con endpoint JSON (sin Twilio)
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from agent import build_graph
from rag import setup_chroma
import redis
import json
from typing import List, Dict

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")



if not api_key:
    raise ValueError("Configura GEMINI_API_KEY en .env")

# Configura Redis (esto se mantiene igual para la memoria)
try:
    # Lee la URL única desde el .env
    redis_connection_url = os.getenv("REDIS_URL")
    if not redis_connection_url:
        raise ValueError("No se encontró la variable de entorno REDIS_URL")

    print(f"Conectando a Redis usando URL...")
    
    redis_client = redis.from_url(
        redis_connection_url,
        decode_responses=True # Mantenemos esto
    )
    
    redis_client.ping()
    print("Conexión a Redis exitosa.")
except (redis.exceptions.ConnectionError, ValueError) as e:
    print(f"Error al conectar a Redis: {e}")
    raise e

# Carga RAG (ejecuta una vez en startup)
document_path = "faqs.txt" # Asegúrate que este sea el path correcto
retriever = setup_chroma(document_path)
graph = build_graph(retriever, api_key)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ajusta esto para tus dominios de producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Modelos Pydantic para el nuevo endpoint ---

class ChatRequest(BaseModel):
    user_id: str  # ID único del usuario (puede ser un email, username, o un ID de sesión)
    message: str  # El mensaje del usuario

class ChatResponse(BaseModel):
    response: str
    user_id: str
    history_length: int

# --- Nuevo Endpoint JSON para el Chat ---

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint para chatear con el agente.
    Recibe un user_id y un mensaje, y usa Redis para mantener el historial.
    """
    try:
        user_id = request.user_id
        body = request.message
        
        print(f"Processing message from {user_id}: {body}")
        
        # Carga estado de conversación de Redis
        state_key = f"user_session:{user_id}" # Nueva clave de estado
        state_str = redis_client.get(state_key)
        state = json.loads(state_str) if state_str else {"messages": []}
        print(f"Loaded state: {state}")
        
        # Convierte mensajes almacenados en objetos LangChain
        messages = []
        for msg in state["messages"]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        # Agrega el nuevo mensaje del usuario
        messages.append(HumanMessage(content=body))
        
        # Actualiza el estado para Redis (solo los mensajes)
        # Esto es temporal por si LangGraph falla, pero se sobrescribe después
        state_for_graph = {"messages": messages}
        
        # Procesa con LangGraph
        response_content = ""
        final_state = {}
        for updated_state in graph.stream(state_for_graph, stream_mode="values"):
            last_message = updated_state["messages"][-1]
            response_content = last_message.content
            final_state = updated_state
        
        print(f"Generated response: {response_content}")
        
        # Convierte el estado final (objetos BaseMessage) a formato serializable (dict)
        serializable_messages = [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
            for m in final_state["messages"]
        ]
        
        # Guarda el estado actualizado en Redis con TTL de 24 horas
        # Usamos `ensure_ascii=False` para guardar correctamente acentos y 'ñ'
        redis_client.set(
            state_key, 
            json.dumps({"messages": serializable_messages}, ensure_ascii=False), 
            ex=86400
        )
        print(f"Saved state to Redis (key: {state_key})")
        
        # Devuelve la respuesta JSON
        return ChatResponse(
            response=response_content,
            user_id=user_id,
            history_length=len(serializable_messages)
        )
    
    except Exception as e:
        print(f"Error en /chat: {str(e)}")
        # Devuelve un error 500 en formato JSON
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal error: {str(e)}"}
        )

# --- Endpoint de Bienvenida (Opcional) ---

@app.get("/")
def read_root():
    return {"message": "Agente de FAQs de Banorte - API"}