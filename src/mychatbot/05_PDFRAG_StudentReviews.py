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
# Upload to VectorDB

def load_qdrant(collection_name):
    client = QdrantClient(path=QDRANT_PATH)

    # すべてのコレクション名を取得
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    # コレクションが存在しなければ作成
    if collection_name not in collection_names:
        # コレクションが存在しない場合、新しく作成します
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print('collection created')

    #return Qdrant(
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name, 
        embedding=OpenAIEmbeddings()
    )

def get_pdf_text(file_uploader_key):
    uploaded_file = st.file_uploader(
        label='Upload your PDF here😇',
        type='pdf',
        accept_multiple_files=False,
        key=file_uploader_key
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

#def build_vector_store(qdrant, pdf_text, metadatas=None):   # metadata: List[dict]
#    # qdrant = load_qdrant()
#    qdrant.add_texts(pdf_text, metadatas=metadatas)         # metadata: List[dict]
#
#    # 以下のようにもできる。この場合は毎回ベクトルDBが初期化される
#    # LangChain の Document Loader を利用した場合は `from_documents` にする
#    # Qdrant.from_texts(
#    #     pdf_text,
#    #     OpenAIEmbeddings(),
#    #     path="./local_qdrant",
#    #     collection_name="my_documents",
#    # )

def page_upload_and_build_vector_db():
    st.title("Upload to VectorDB")
    qdrant = load_qdrant(collection_name=COLLECTION_NAME)

    container_Syllabus = st.container()
    container_StudentHandbook = st.container()
    

    with container_Syllabus:
        st.markdown("### Syllabus")
        pdf_text = get_pdf_text("Syllabus")
        
        col_lectureCode, col_uploadButton = st.columns((5, 1), vertical_alignment="bottom")
        with col_lectureCode:
            lectureCode = st.text_input("Lecture Code")
        with col_uploadButton:
            if st.button("Upload"):
                if not pdf_text:
                    st.warning("Upload Syllabus")
                elif not lectureCode:
                    st.warning("Fill in the blank")
                else:    
                    with st.spinner("Loading ..."):
                        qdrant.add_texts(pdf_text, metadatas=[{"type": "Syllabus", "Lecture Code": lectureCode}])
                        #st.rerun()
                    
    with container_StudentHandbook:
        st.markdown("### Student Handbook")
        pdf_text = get_pdf_text("Student Handbook")

        #if st.button("Upload") and pdf_text:
        #    with st.spinner("Loading ..."):
        #        #build_vector_store(qdrant, pdf_text)
        #        qdrant.add_texts(pdf_text, metadatas=[{"type": "Syllabus"}])
        #        #st.rerun()

        col_organization, col_uploadButton = st.columns((5, 1), vertical_alignment="bottom")
        with col_organization:
            organization = st.text_input("Organization")
        with col_uploadButton:
            if st.button("Upload", key="uploadStudentHandbook"):
                if not pdf_text:
                    st.warning("Upload")
                elif not organization:
                    st.warning("Fill in the blank")
                else:    
                    with st.spinner("Loading ..."):
                        qdrant.add_texts(pdf_text, metadatas=[{"type": "Student Handbook", "Organization": organization} for _ in range(len(pdf_text))])



# -------------------------------------------------------------------------------------------------------
# Manage VectorDB

def filter_record(record, selected_type, selected_):
    if record.payload["metadata"] == None:
        return True
    if selected_type != "ALL" and record.payload["metadata"]["type"] != selected_type:
        return False
    if selected_ != "ALL" and record.payload["metadata"]["type"] != selected_:
        return False
    return True


def page_manage_vector_db():
    st.title("Manage VectorDB")
    container_manager = st.container()
    qdrant = load_qdrant(collection_name=COLLECTION_NAME)
    # Manager
    if qdrant:
        with container_manager:
            st.markdown("## Select")
            record_list = qdrant.client.scroll(COLLECTION_NAME, limit=200)

            if not record_list:
                st.warning("No Data")
            else:
                # for record in record_list[0]:
                #     st.write(f"{record.id}, {record.payload["page_content"][:50]}...")

                selected_records = st.multiselect(
                    "Select Records you wanna delete", 
                    [str(record.payload["page_content"][:100]) + "...\n" + str(record.id) for record in record_list[0]],
                )

                if selected_records:
                    for record in selected_records:
                        st.write(record)
                
                selected_ids = [selected_record[-32:] for selected_record in selected_records]

                # Delete Button
                if st.button("Delete"):
                    qdrant.client.delete(collection_name=COLLECTION_NAME, points_selector=selected_ids)
                    st.success(f"{selected_ids} deleted.")
                    st.rerun()

                # List
                st.markdown("## List")    
                # フィルタ機能
                col_type, col_ = st.columns(2)
                with col_type:
                    selected_type = st.selectbox("Type", ["ALL", "Syllabus", "Student Handbook", "Student Review"])
                with col_:
                    selected_ = st.selectbox("_", ["ALL", "_"])
                filtered_record_list = [record for record in record_list[0] if filter_record(record, selected_type, selected_)]
                st.write(filtered_record_list)
                #st.write(record_list[0])

                if st.button("DeleteALL"):
                    for record in filtered_record_list:
                        qdrant.client.delete(collection_name=COLLECTION_NAME, points_selector=[str(record.id)])
                    st.rerun()
                
                # id による参照
                #st.write(qdrant.client.retrieve(collection_name=COLLECTION_NAME, ids=[selected]).payload["page_content"])


# -------------------------------------------------------------------------------------------------------
# Student Reviews
# メモ：大学・学部・学科・科目・出席点・授業の雰囲気・テストの有無と難易度・充実度（5段階評価）・学んだこと等に関する感想

def filter_review(review, selected_faculty, selected_major):
    if review.payload["metadata"] == None:
        return False
    if "type" not in review.payload["metadata"]:
        return False
    if review.payload["metadata"]["type"] != "Student Review":
        return False
    if selected_faculty != "ALL" and ("faculty" in review.payload["metadata"]) and review.payload["metadata"]["faculty"] != selected_faculty:
        return False
    if selected_major != "ALL" and ("major" in review.payload["metadata"]) and review.payload["metadata"]["major"] != selected_major:
        return False
    return True

def page_student_reviews():
    st.title("Student Reviews")

    qdrant = load_qdrant(collection_name=COLLECTION_NAME)
    
    # sidebar
    st.sidebar.markdown("")
    st.sidebar.markdown("")
    st.sidebar.markdown("### Post Your Review!")
    college = st.sidebar.text_input("College / University")
    faculty = st.sidebar.text_input("Faculty")
    major = st.sidebar.text_input("Major")
    st.sidebar.markdown("")
    st.sidebar.markdown("")
    subject = st.sidebar.text_input("Subject")
    lectureCode = st.sidebar.text_input("Lecture Code")
    attendance_points = st.sidebar.selectbox("Attendance Points", ["O", "X"])
    tests = st.sidebar.text_area("Tests")
    comment = st.sidebar.text_area("Comment")
    satisfaction = st.sidebar.select_slider("Satisfaction", ["1", "2", "3", "4", "5"])
    ## 格納
    review_text = f"科目名(授業コード)：{subject} ({lectureCode})<br>出席点の有無：{"あり" if (attendance_points=="O") else "なし"}<br>テストについて：{tests}<br>感想やコメント：{comment}<br>満足度：{satisfaction}"
    if st.sidebar.button("Post", key="postStudentReview", use_container_width=True):
        # レビューを投稿（ベクトルDBに格納）
        with st.spinner("Loading ..."):
            qdrant.add_texts([review_text], metadatas=[{"type": "Student Review", "Subject": subject, "Lecture Code": lectureCode, "college": college, "faculty": faculty, "major": major}])

    # 本体
    container_reviewList = st.container()

    reviews_list = qdrant.client.scroll(COLLECTION_NAME, limit=200)
    st.markdown(
        """
        <style>
        .custom-container {
            background-color: #e8e8e8;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-size: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    with container_reviewList:
        # フィルタ機能
        #col_college, col_faculty, col_major = st.columns(3)
        #with col_college:
        #    selected_college = st.selectbox("Type", ["ALL", "Kobe", ""])
        col_faculty, col_major, col_subject, col_lectureCode = st.columns((3,3,2,2))
        with col_faculty:
            selected_faculty = st.selectbox("Faculty", ["ALL", "Engineering"])
        with col_major:
            selected_major = st.selectbox("Major", ["ALL", "Information", "Mechanical"])
        with col_subject:
            inputed_subject = st.text_input("Subject", key="subject")
        with col_lectureCode:
            inputed_lectureCode = st.text_input("Lecture Code", key="lectureCode")
        filtered_reviews = [review for review in reviews_list[0] if filter_review(review, selected_faculty, selected_major)]
        if inputed_subject:
            filtered_reviews = [review for review in filtered_reviews if (review.payload["metadata"]["Subject"] == inputed_subject)]
        if inputed_lectureCode:
            filtered_reviews = [review for review in filtered_reviews if (review.payload["metadata"]["Lecture Code"] == inputed_lectureCode)]
        
        # 表示
        col_left, col_center, col_right = st.columns(3)
        col_list = [col_left, col_center, col_right]
        for i, review in enumerate(filtered_reviews):
            with col_list[i % 3]:
                with st.container():
                    st.markdown(f'<div class="custom-container"> {review.payload["page_content"]} </div>', unsafe_allow_html=True)
                

        #for review in filtered_reviews:
        #    st.write(review.payload["page_content"], unsafe_allow_html=True)


# -------------------------------------------------------------------------------------------------------
# Ask

def build_qa_model(llm):
    qdrant = load_qdrant(collection_name=COLLECTION_NAME)
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
            SystemMessage(content="Let's decide the timetable together!")
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

    st.sidebar.markdown("## Credit")
    st.sidebar.markdown(f"基礎教養科目：      {2}  / {4}")
    st.sidebar.markdown(f"総合教養科目：      {3}  / {4}")
    st.sidebar.markdown(f"専門科目：　　      {70} / {80}")

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

    with QA_container:
        init_messages()
        st.markdown("## Ask LLM")

        GPT_container, StudentReview_container = st.columns(2)
        with GPT_container:
            st.markdown("##### Syllabus")
            # ユーザーの入力を監視
            if user_input := st.chat_input("例）授業内容を箇条書きで教えて！"):
                st.session_state.messages.append(HumanMessage(content=user_input))
                #with st.spinner("ChatGPT is typing ..."):
                #    answer, cost = get_answer(llm, st.session_state.messages)
                qa = build_qa_model(llm)
                if qa:
                    with st.spinner("ChatGPT is typing ..."):
                        answer, cost = ask(qa, user_input)
                    #st.write(answer)
                    st.session_state.messages.append(AIMessage(content=answer["result"]))
                    st.session_state.costs.append(cost)
                else:
                    answer = None
            messages = st.session_state.get('messages', [])
            for message in reversed(messages):
                if isinstance(message, AIMessage):
                    with st.chat_message('assistant', avatar="🧠"):
                        st.markdown(message.content)
                elif isinstance(message, HumanMessage):
                    with st.chat_message('user', avatar="😀"):
                        st.markdown(message.content)
                else:  # isinstance(message, SystemMessage):
                    st.write(f"System message: {message.content}")
        with StudentReview_container:
            # TODO: 口コミをここで検索
            st.markdown("##### Student Reviews")
            st.markdown("（左と同じように実装予定。。）")



# ------------------------------------------------------------------------------------------------------

def main():
    init_page()

    switch = st.sidebar.checkbox("Admin")

    selection_admin = st.sidebar.radio("Admin", ["Upload to VectorDB", "Manage VectorDB"], disabled=not switch)
    if switch:
        if selection_admin == "Upload to VectorDB":
            page_upload_and_build_vector_db()
        elif selection_admin == "Manage VectorDB":
            page_manage_vector_db()

    selection_user = st.sidebar.radio("Menu", ["Student Reviews", "Ask LLM"], disabled=switch)
    if not switch:
        if selection_user == "Student Reviews":
            page_student_reviews()
        elif selection_user == "Ask LLM":
            page_ask_my_pdf()

    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")


if __name__ == '__main__':
    main()