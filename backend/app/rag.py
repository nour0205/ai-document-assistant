import numpy as np
from openai import OpenAI
from . import data_store



def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def answer_question(question: str) -> str:
    client = OpenAI()

    # 1. Embed the question
    question_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    # 2. Compute similarity scores
    scores = []
    for chunk, emb in zip(data_store.chunks, data_store.embeddings):

        score = cosine_similarity(question_embedding, emb)
        scores.append((chunk, score))

    # 3. Select top-k chunks
    scores.sort(key=lambda x: x[1], reverse=True)
    top_k = 8
    top_chunks = [chunk for chunk, _ in scores[:top_k]]

    # 4. Build context
    context = "\n\n---\n\n".join(top_chunks)

    # 5. Build prompt
    prompt = f"""
You are an AI assistant that answers questions using ONLY the provided context.
Do not use any outside knowledge.
If the answer is not supported by the context, say:
"I don't know based on the document."

Context:
{context}

Question:
{question}

Answer:
"""

    # 6. Generate answer
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content
