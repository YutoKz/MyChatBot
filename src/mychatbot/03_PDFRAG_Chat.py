# pip install pycryptodome
from glob import glob
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain_community.callbacks.manager import get_openai_callback

from PyPDF2 import PdfReader
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore  # Qdrant
from langchain.chains import RetrievalQA

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

import pandas as pd

QDRANT_PATH = "./local_qdrant"
COLLECTION_NAME = "my_collection_2"

DATA_PATH = "./data"


def init_page():
    st.set_page_config(
        page_title="Ask My PDF(s)",
        page_icon="🤗"
    )
    st.sidebar.title("Nav")
    st.session_state.costs = []


def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5-turbo", "GPT-4"))
    if model == "GPT-3.5-turbo":
        st.session_state.model_name = "gpt-3.5-turbo"
    else: # GPT-4
        st.session_state.model_name = "gpt-4"
    
    # 300: 本文以外の指示のトークン数 (以下同じ)
    st.session_state.max_token = OpenAI.modelname_to_contextsize(st.session_state.model_name) - 300
    return ChatOpenAI(temperature=0, model=st.session_state.model_name)

# -------------------------------------------------------------------------------------------------------
# VectorDB

def get_pdf_text():
    uploaded_file = st.file_uploader(
        label='Upload your PDF here😇',
        type='pdf',
        accept_multiple_files=False,
    )
    if uploaded_file:
        pdf_reader = PdfReader(uploaded_file)
        text = '\n\n'.join([page.extract_text() for page in pdf_reader.pages])
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="text-embedding-ada-002",
            # 適切な chunk size は質問対象のPDFによって変わるため調整が必要
            # 大きくしすぎると質問回答時に色々な箇所の情報を参照することができない
            # 逆に小さすぎると一つのchunkに十分なサイズの文脈が入らない
            chunk_size=500,
            chunk_overlap=0,
        )
        return text_splitter.split_text(text)
    else:
        return None


def load_qdrant():
    client = QdrantClient(path=QDRANT_PATH)

    # すべてのコレクション名を取得
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    # コレクションが存在しなければ作成
    if COLLECTION_NAME not in collection_names:
        # コレクションが存在しない場合、新しく作成します
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print('collection created')

    #return Qdrant(
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME, 
        embedding=OpenAIEmbeddings()
    )


def build_vector_store(qdrant,pdf_text):
    # qdrant = load_qdrant()
    qdrant.add_texts(pdf_text)

    # 以下のようにもできる。この場合は毎回ベクトルDBが初期化される
    # LangChain の Document Loader を利用した場合は `from_documents` にする
    # Qdrant.from_texts(
    #     pdf_text,
    #     OpenAIEmbeddings(),
    #     path="./local_qdrant",
    #     collection_name="my_documents",
    # )

def page_pdf_upload_and_build_vector_db():
    st.title("VectorDB")
    container = st.container()
    container_manager = st.container()

    qdrant = load_qdrant()
    
    with container:
        st.markdown("## Upload")
        pdf_text = get_pdf_text()
        if pdf_text:
            with st.spinner("Loading PDF ..."):
                build_vector_store(qdrant, pdf_text)

    # Manager
    if qdrant:
        with container_manager:
            st.markdown("## Manage")
            record_list = qdrant.client.scroll(COLLECTION_NAME, limit=200)

            if not record_list:
                st.warning("No Data")
            else:
                # for record in record_list[0]:
                #     st.write(f"{record.id}, {record.payload["page_content"][:50]}...")

                selected = st.selectbox("Select ID you wanna delete", [str(record.payload["page_content"][:100]) + "\n" + str(record.id) for record in record_list[0]])

                # Delete Button
                if st.button("Delete"):
                    qdrant.client.delete(collection_name=COLLECTION_NAME, points_selector=[selected[-32:]])
                    st.success(f"ID: {selected[-32:]} deleted.")
                if st.button("DeleteALL"):
                    for record in record_list[0]:
                        qdrant.client.delete(collection_name=COLLECTION_NAME, points_selector=[str(record.id)])


# -------------------------------------------------------------------------------------------------------
# Ask

def build_qa_model(llm):
    qdrant = load_qdrant()
    retriever = qdrant.as_retriever(
        # "mmr",  "similarity_score_threshold" などもある
        search_type="similarity",
        # 文書を何個取得するか (default: 4)
        search_kwargs={"k":10}
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )

def ask(qa, query):
    with get_openai_callback() as cb:
        # query / result / source_documents
        answer = qa(query)

    return answer, cb.total_cost

def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]
        st.session_state.costs = []

def get_answer(llm, messages):
    with get_openai_callback() as cb:
        answer = llm(messages)
    return answer.content, cb.total_cost


def page_ask_my_pdf():
    st.title("Ask")

    llm = select_model()
    timetable_container = st.container()
    QA_container = st.container()
    #query_container = st.container()
    #response_container = st.container()

    with timetable_container:
        st.markdown("## Timetable")
        # 学部, 学科, 学年, 前期後期
        col_faculty, col_major, col_grade, col_semester = st.columns(4)
        with col_faculty:
            faculty = st.selectbox("Faculty", ["Engineering"])
        with col_major:
            major = st.selectbox("major", ["Information"])
        with col_grade:
            grade = st.selectbox("grade", ["1", "2", "3", "4"])
        with col_semester:
            semester = st.selectbox("semester", ["First", "Second"])
        
        # csvの表示
        df_timetable = pd.read_csv(DATA_PATH +"/timetable/"+ faculty+"_"+major+"_"+grade+"_"+semester +".csv")
        
        st.markdown(
            """
                <style>
                .kiso-kyouyou {
                    color: lightgreen;  /* 文字の色 */
                    font-weight: bold;
                }
                .sougou-kyouyou {
                    color: orange;  /* 文字の色 */
                    font-weight: bold;
                }
                </style>
            """, 
            unsafe_allow_html=True
        )
        # セルごとに書式を変更
        # df.at[0, '名前'] = '<span class="highlight-cell">田中</span>'

        ## Pandas Stylerで文字の大きさを一括指定
        df_timetable = df_timetable.style.set_properties(**{
            'font-size': '12px',
            'text-align': 'center',
        }).set_table_styles([
                {'selector': 'th', 'props': [('min-width', '141px'), ('max-width', '141px')]},  # ヘッダーの列幅
                {'selector': 'td', 'props': [('min-width', '141px'), ('max-width', '141px')]}
            ]  # 各セルの列幅
        ).hide()


        st.markdown(df_timetable.to_html(escape=False, bold_headers=True), unsafe_allow_html=True)


    #with query_container:
    #    st.markdown("## Query")
    #    query = st.text_input("Query: ", key="input")
    #    if not query:
    #        answer = None
    #    else:
    #        qa = build_qa_model(llm)
    #        if qa:
    #            with st.spinner("ChatGPT is typing ..."):
    #                answer, cost = ask(qa, query)
    #            st.session_state.costs.append(cost)
    #        else:
    #            answer = None
    #
    #    #if answer:
    #    #    with response_container:
    #    #        st.markdown("## Answer")
    #    #        st.write(answer)
    #with response_container:
    #    st.markdown("## Answer")
    #    if answer:
    #        st.markdown("##### result")
    #        st.write(answer["result"])
    #        st.markdown("##### source_documents")
    #        for i in range(len(answer["source_documents"])):
    #            st.markdown(f"{i+1}.")
    #            st.write(answer["source_documents"][i].page_content)

    with QA_container:
        init_messages()

        # ユーザーの入力を監視
        if user_input := st.chat_input("ChatGPTと相談しよう!!"):
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("ChatGPT is typing ..."):
                answer, cost = get_answer(llm, st.session_state.messages)
            st.session_state.messages.append(AIMessage(content=answer))
            st.session_state.costs.append(cost)

        messages = st.session_state.get('messages', [])
        for message in messages:
            if isinstance(message, AIMessage):
                with st.chat_message('assistant'):
                    st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message('user'):
                    st.markdown(message.content)
            else:  # isinstance(message, SystemMessage):
                st.write(f"System message: {message.content}")












# ------------------------------------------------------------------------------------------------------

def main():
    init_page()

    selection = st.sidebar.radio("Go to", ["VectorDB", "Ask"])
    if selection == "VectorDB":
        page_pdf_upload_and_build_vector_db()
    elif selection == "Ask":
        page_ask_my_pdf()

    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")


if __name__ == '__main__':
    main()