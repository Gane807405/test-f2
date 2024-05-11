from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit.components as components
from langchain_community.llms import Ollama
load_dotenv()
base_dir = os.path.dirname(os.path.abspath(__file__))
excel_file_path1 = os.path.join(base_dir, os.getenv("EXCEL_FILE_PATH"))
faiss_index_path = os.path.join(base_dir, os.getenv("FAISS_INDEX"))
llm = Ollama(model="mistral")
st.title('Instagram Influencer Search')
model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)










db = FAISS.load_local(faiss_index_path, embeddings,
                      allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 15})


def format_docs(docs):
    return "\n\n".join(f"{doc.page_content}\n\nMetadata: {doc.metadata}" for doc in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            """
            Your goal is to recommend influencer/influencer media to users based on their 
            query and the retrieved context. If a retrieved influencer/media doesn't seem 
            relevant, omit it from your response. If your context is empty
            or none of the retrieved results are relevant, do not recommend any
            , but instead tell the user you couldn't find any influencers or media
            that match their query. Aim for ten to fifteen media recommendations,
            as long as they are relevant. Your recommendation should be relevant, 
            original, and at least two to three sentences long.
            
            YOU CANNOT RECOMMEND A MEDIA/INFLUENCER IF IT DOES NOT APPEAR IN YOUR 
            CONTEXT.

            # TEMPLATE FOR OUTPUT
            - **Caption**:
                - Influencer_ID:
                - Media_ID:
                - (Your reasoning for recommending this media)
            
            Question: {question} 
            Context: {context} 
            """
        ),
    ]
)

rag_chain_from_docs = (
    RunnablePassthrough.assign(
        context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)


def process_llm_output(query):
    output = {}
    curr_key = None
    results = []

    for chunk in rag_chain_with_source.stream(query):
        for key in chunk:
            if key not in output:
                output[key] = chunk[key]
            else:
                output[key] += chunk[key]
            if key != curr_key:
                print(f"\n\n{key}: {chunk[key]}", end="", flush=True)
            else:
                print(chunk[key], end="", flush=True)
            curr_key = key

            if key == 'context':
                for doc in chunk[key]:
                    influencer_id = doc.metadata.get('influencer_id', '')
                    instagram_media_id = doc.metadata.get('instagram_media_id', '')
                    results.append({'influencer_id': str(influencer_id), 'instagram_media_id': instagram_media_id})

    output['results'] = results
    print("this is the results", results)
    return results


def process_results(query):
    results = process_llm_output(query)
    excel_file_path = excel_file_path1
    df = pd.read_excel(excel_file_path)
    grouped_results = {}
    for result in results:
        influencer_id = result["influencer_id"]
        media_id = result["instagram_media_id"]
        if influencer_id in grouped_results:
            grouped_results[influencer_id].append(media_id)
        else:
            grouped_results[influencer_id] = [media_id]
    return grouped_results, df


with st.form("my_form"):
    query = st.text_input('Enter your search query: ')
    
    if st.form_submit_button("Search"):
        
        grouped_results, df = process_results(query)
        search_results = grouped_results.get(query, [])
        st.header("Search Results")
        for influencer_id, media_ids in grouped_results.items():
            if not search_results or influencer_id in search_results:
                st.markdown("_______")
                st.markdown(f"**Influencer ID:** {influencer_id}")
                for media_id in media_ids:
                    media_info_row = df[(df['influencer_id'].astype(str) == influencer_id) & (df['instagram_media_id'].astype(str) == media_id)]
                    if not media_info_row.empty:
                        media_url = media_info_row['url'].iloc[0]

                        st.markdown(f"**Instagram Media ID:** {media_id}")

                        st.markdown(f"**Media URL:** {media_url}")
                        if media_url:
                            components.v1.html(f'<blockquote class="instagram-media" data-instgrm-permalink="{media_url}" data-instgrm-version="13"></blockquote><script async src="//www.instagram.com/embed.js"></script>', width=400, height=400, scrolling=True)
                    else:
                        st.markdown("not found matching in excel ")
