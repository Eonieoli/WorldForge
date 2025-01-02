# WorldForge/src/backend/app/main.py

import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import ChatUpstage
from langchain_upstage import UpstageEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel
from mangum import Mangum

load_dotenv()

# upstage models
chat_upstage = ChatUpstage(model="solar-pro")
embedding_upstage = UpstageEmbeddings(model="embedding-query")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "worldforge"

# create new index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=4096,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

pinecone_vectorstore = PineconeVectorStore(index=pc.Index(index_name), embedding=embedding_upstage)

pinecone_retriever = pinecone_vectorstore.as_retriever(
    search_type='mmr',  # default : similarity(유사도) / mmr 알고리즘
    search_kwargs={"k": 6}  # 쿼리와 관련된 chunk를 6개 검색하기 (default : 4)
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class AssistantRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


class ChatRequest(BaseModel):
    messages: List[ChatMessage]  # Entire conversation for naive mode


class MessageRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(req: MessageRequest):
    # Retrieve context from Pinecone
    retrieved_docs = pinecone_retriever.retrieve(req.message)
    context = "\n".join(doc["text"] for doc in retrieved_docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                당신은 영화, 드라마 등의 콘텐츠를 모두 아는 콘텐츠의 신입니다. 유명한 작가가 되고자하는 열정있는 사람들이 당신에게 질문 할 예정입니다.
                친절하게 당신이 아는 모든 것을 말해주세요. 단 모르는게 있으면 모른다고 답하고, 작가의 질문에 오타를 고려하여 비슷한 질문이 있다면 이를 수정해서 질문의 의도나 
                오타를 수정하여 다시 물어봐주세요. 또한 비슷한 정보가 있으면 vectorstore에 있는 정보를 우선시 해서 답변해주세요
                ---
                CONTEXT:
                {context}
                """,
            ),
            ("human", "{input}"),
        ]
    )

    chain = prompt | chat_upstage | StrOutputParser()

    # Use the custom chain with the retrieved context
    response = chain.invoke({"input": req.message, "context": context})

    return {"reply": response}


@app.get("/health")
@app.get("/")
async def health_check():
    return {"status": "ok"}

handler = Mangum(app)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
