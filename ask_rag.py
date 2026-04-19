import requests

BASE_URL = "http://localhost:8000"


def ingest_documents():
    response = requests.post(f"{BASE_URL}/ingest")
    response.raise_for_status()
    return response.json()


def ask_question(question, chat_history=None, top_k=3, topic_filter=None):
    payload = {
        "question": question,
        "chat_history": chat_history or [],
        "top_k": top_k,
        "topic_filter": topic_filter,
    }
    response = requests.post(f"{BASE_URL}/ask", json=payload)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    print("Ingesting documents...")
    ingest_result = ingest_documents()
    print(ingest_result)

    question = "What is retrieval-augmented generation?"
    print(f"\nAsking: {question}")
    answer_result = ask_question(question)
    print(answer_result)
