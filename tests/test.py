
# 약 1분 소요

import os
import getpass
import warnings

warnings.filterwarnings("ignore")

# Get the Upstage API key using getpass
try:
    if "UPSTAGE_API_KEY" not in os.environ or not os.environ["UPSTAGE_API_KEY"]:
        os.environ["UPSTAGE_API_KEY"] = "up_OCwdy6zbYwSmBqGvfuOBck87rEAOS"

    print("API key has been set successfully.")

except:
    print("Something wrong with your API KEY. Check your API Console again.")

# Document Parse로 다운로드 된 문서 불러오기

from langchain_upstage import UpstageDocumentParseLoader
files = {"wonka.docx", "hell2.docx","insideout2.docx"}

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings
from langchain.docstore.document import Document

# UpstageDocumentParseLoader 클래스가 이미 정의되어 있다고 가정합니다.
for file in files:
    print(f"Processing file: {file}")

    # Step 1: Load document
    layzer = UpstageDocumentParseLoader(
        file,  # 현재 파일을 불러오기
        output_format='html',  # 결과물 형태: HTML
        coordinates=False  # 이미지 OCR 좌표계 사용 안 함
    )
    docs = layzer.load()
    

    # Step 3: Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(docs)
    print(f"Splits for {file}: {len(splits)}")

    # Step 4: Embed & Index
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=UpstageEmbeddings(model="solar-embedding-1-large")
    )
    print(f"Embedding and indexing completed for {file}.\n")





from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatUpstage(model="solar-pro")

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

chain = prompt | llm | StrOutputParser()

query = "영화 지옥2에 나오는 등장인물 다 알려줘. 연기한 배우랑 함께"
# Dense Retriever 생성
retriever = vectorstore.as_retriever(
    search_type= 'mmr', # default : similarity(유사도) / mmr 알고리즘
    search_kwargs={"k": 6} # 쿼리와 관련된 chunk를 3개 검색하기 (default : 4)
)

result_docs = retriever.invoke(query) # 쿼리 호출하여 retriever로 검색


response = chain.invoke({"context": result_docs, "input": query})

print(response)