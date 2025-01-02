# WorldForge/src/backend/app/embed.py

import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import UpstageDocumentParseLoader
from langchain_upstage import UpstageEmbeddings
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# upstage models
embedding_upstage = UpstageEmbeddings(model="embedding-query")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "worldforge"
pdf_paths = ["wonka.pdf", "hell2.pdf", "insideout2.pdf"]  # List of PDF file paths

# create new index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=4096,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-2"),
    )

print("start")

for pdf_path in pdf_paths:
    print(f"Processing {pdf_path}...")
    # Parse the PDF document
    document_parse_loader = UpstageDocumentParseLoader(
        pdf_path,
        output_format='html',  # 결과물 형태 : HTML
        coordinates=False  # 이미지 OCR 좌표계 가져오지 않기
    )
    docs = document_parse_loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )
    splits = text_splitter.split_documents(docs)

    # Store the embeddings in Pinecone
    PineconeVectorStore.from_documents(
        splits, embedding_upstage, index_name=index_name
    )

print("end")
