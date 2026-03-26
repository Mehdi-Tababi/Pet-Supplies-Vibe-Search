import streamlit as st
import pandas as pd
import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from chromadb.utils import embedding_functions
import google.generativeai as genai


@st.cache_data(ttl=3600, show_spinner=False)
def pick_gemini_model(api_key):
    """Pick an available Gemini model that supports generateContent."""
    genai.configure(api_key=api_key)
    preferred_models = [
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]
    fallback = "gemini-1.5-flash"

    try:
        available = []
        for model in genai.list_models():
            methods = getattr(model, "supported_generation_methods", [])
            if "generateContent" in methods:
                available.append(model.name.replace("models/", ""))

        for name in preferred_models:
            if name in available:
                return name

        if available:
            return sorted(available)[0]
    except Exception:
        pass

    return fallback


@st.cache_resource
def load_search_system():
    df = pd.read_csv("cleaned_pet_data.csv")
    df["parent_asin"] = df["parent_asin"].astype(str)
    df_by_asin = df.set_index("parent_asin", drop=False)
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    chroma_client = chromadb.PersistentClient(path="./pet_supplies_db")
    collection = chroma_client.get_collection(name="pet_vibe_search", embedding_function=sentence_transformer_ef)

    documents_list = df["vibe_text"].tolist()
    ids_list = df["parent_asin"].tolist()
    tokenized_corpus = [str(doc).lower().split(" ") for doc in documents_list]
    bm25 = BM25Okapi(tokenized_corpus)

    return df_by_asin, collection, bm25, ids_list


df_by_asin, collection, bm25_engine, ids_list = load_search_system()


def weighted_hybrid_search(user_query, top_n=5, alpha=0.5):
    tokenized_query = user_query.lower().split(" ")
    bm25_scores = bm25_engine.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:20]
    keyword_ranked_ids = [ids_list[i] for i in top_bm25_indices]

    vector_results = collection.query(query_texts=[user_query], n_results=20)
    vector_ranked_ids = vector_results["ids"][0]

    rrf_scores = {}
    k_constant = 60

    for rank, doc_id in enumerate(vector_ranked_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + ((1.0 / (k_constant + rank + 1)) * alpha)
    for rank, doc_id in enumerate(keyword_ranked_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + ((1.0 / (k_constant + rank + 1)) * (1.0 - alpha))

    final_sorted_results = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_id for doc_id, score in final_sorted_results[:top_n]]


st.set_page_config(page_title="AI Pet Assistant", page_icon="🐾", layout="wide")

with st.sidebar:
    st.header("⚙️ App Settings")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    st.markdown("Get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey)")
    alpha_dial = st.slider("Search Tuning (Alpha)", 0.0, 1.0, 0.5, 0.1, help="0.0 = BM25, 1.0 = Vector")

st.title("🐾 AI Pet Shopping Assistant")
st.write("Ask me anything! I will search the database and give you a custom recommendation.")

user_query = st.text_input("What does your pet need today?")

if st.button("Search & Ask AI", type="primary"):
    if not user_query:
        st.warning("Please enter a search query.")
    else:
        with st.spinner("Searching database and thinking..."):
            top_ids = weighted_hybrid_search(user_query, top_n=5, alpha=alpha_dial)

            context_data = ""
            for target_id in top_ids:
                row = df_by_asin.loc[target_id]
                context_data += f"\n- Product: {row['title']} (Brand: {row.get('brand', 'Unknown')})\n  Details: {row['vibe_text'][:400]}...\n"

            ai_text = None
            ai_error = None

            if api_key:
                system_prompt = f"""
                You are a helpful and expert pet store assistant.
                A customer asked: \"{user_query}\"

                Based ONLY on the following products from our database, write a friendly,
                short recommendation explaining which product is best for them and why based on the vibes/reviews.

                Database Products:
                {context_data}
                """

                try:
                    model_name = pick_gemini_model(api_key)
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(system_prompt)
                    ai_text = getattr(response, "text", "No text response received.")
                except Exception as e:
                    ai_error = str(e)

            if ai_text:
                st.success("✨ AI Recommendation")
                st.write(ai_text)
            elif api_key:
                st.warning("AI recommendation is temporarily unavailable.")
                if ai_error and "429" in ai_error:
                    st.info("Gemini API quota exceeded. Wait for reset or enable billing.")
                elif ai_error and "404" in ai_error:
                    st.info("Selected Gemini model is unavailable for your API version/project.")
                elif ai_error:
                    st.info(f"Gemini request failed: {ai_error}")
                st.markdown("Showing best matching products from hybrid search only.")
            else:
                st.info("No API key provided. Showing search results without AI summary.")

            st.divider()
            st.subheader("📚 Recommended Products")
            for i, target_id in enumerate(top_ids):
                row = df_by_asin.loc[target_id]
                st.markdown(f"**{i+1}. {row['title']}** ({row.get('brand', 'Unknown')})")
