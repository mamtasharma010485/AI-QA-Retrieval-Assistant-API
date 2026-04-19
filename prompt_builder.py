from typing import List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from app.schemas import Message, DocumentResponse

SYSTEM_TEMPLATE = """You are a helpful AI expert and technical assistant.
Your task is to answer the user's question based strictly on the provided context retrieved from a knowledge base.

Follow these rules:
1. Provide a short summary or direct answer at the beginning.
2. Ensure the answer is grounded ONLY in the retrieved chunks. Read the chunks carefully.
3. If the answer to the question cannot be found in the provided context, clearly state: "The requested information is not available in the retrieved documents." Do not attempt to guess or hallucinate.
4. Keep the response concise, beginner-friendly, and informative.
5. Avoid unsupported claims.

Retrieved Context:
{context}
"""

def build_messages(question: str, chat_history: List[Message], documents: List[DocumentResponse]) -> List[BaseMessage]:
    """Builds the full message list for the chat model."""
    
    # Format the context from retrieved documents
    context_str = "\n\n".join([f"--- Chunk {d.chunk_id} (Topic: {d.topic}) ---\n{d.page_content}" for d in documents])
    
    if not documents:
        context_str = "No relevant documents were retrieved."
        
    system_message = SystemMessage(content=SYSTEM_TEMPLATE.format(context=context_str))
    
    messages: List[BaseMessage] = [system_message]
    
    for msg in chat_history:
        if msg.role.lower() == "user":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role.lower() == "assistant":
            messages.append(AIMessage(content=msg.content))
            
    messages.append(HumanMessage(content=question))
    return messages
