import os
import io
import argparse
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import requests
import torch
import clip
import faiss
from datasets import load_dataset
from pathlib import Path
from PIL import Image
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

#RAG pipeline class
class FashionRAGPipeline:
    def __init__(self, groq_api_key=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API key not found")
        self.groq_client = Groq(api_key=api_key)
        self.cache_dir = Path("./cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.dataset_path = self.cache_dir / "fashion_dataset.pkl"
        self.embeddings_path = self.cache_dir / "embeddings.pkl"
        self.index_path = self.cache_dir / "faiss_index.pkl"
        self.clip_model = None
        self.clip_preprocess = None
        self.dataset = None
        self.embeddings = None
        self.faiss_index = None

    def load_clip_model(self):
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

    def load_dataset(self, limit=1000):
        if self.dataset_path.exists():
            with open(self.dataset_path, "rb") as f:
                self.dataset = pickle.load(f)
        else:
            dataset = load_dataset("tomytjandra/h-and-m-fashion-caption", split="train")
            dataset = dataset.select(range(min(limit, len(dataset))))
            self.dataset = dataset.to_pandas()
            with open(self.dataset_path, "wb") as f:
                pickle.dump(self.dataset, f)

    def generate_embeddings(self):
        if self.embeddings_path.exists():
            with open(self.embeddings_path, "rb") as f:
                self.embeddings = pickle.load(f)
            return

        self.load_clip_model()
        embeddings = []

        for idx, row in self.dataset.iterrows():
            text = row.get("text") or ""
            text_tokens = clip.tokenize([text]).to(self.device)

            with torch.no_grad():
                text_emb = self.clip_model.encode_text(text_tokens).cpu().numpy()[0]

            image_url = row.get("image") or None
            if image_url:
                try:
                    img = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
                    img_tensor = self.clip_preprocess(img).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        img_emb = self.clip_model.encode_image(img_tensor).cpu().numpy()[0]
                except:
                    img_emb = np.zeros_like(text_emb)
            else:
                img_emb = np.zeros_like(text_emb)

            combined_emb = (text_emb + img_emb) / 2
            embeddings.append(combined_emb)

        self.embeddings = np.array(embeddings)
        with open(self.embeddings_path, "wb") as f:
            pickle.dump(self.embeddings, f)

    def build_faiss_index(self):
        if self.index_path.exists():
            with open(self.index_path, "rb") as f:
                self.faiss_index = pickle.load(f)
            return

        if self.embeddings is None:
            self.generate_embeddings()

        faiss.normalize_L2(self.embeddings)
        dim = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.embeddings.astype(np.float32))

        with open(self.index_path, "wb") as f:
            pickle.dump(self.faiss_index, f)

    def search_similar(self, query=None, image=None, top_k=5):
        if self.clip_model is None:
            self.load_clip_model()

        if image is not None:
            img_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                query_emb = self.clip_model.encode_image(img_tensor).cpu().numpy()
        elif query is not None:
            text_tokens = clip.tokenize([query]).to(self.device)
            with torch.no_grad():
                query_emb = self.clip_model.encode_text(text_tokens).cpu().numpy()
        else:
            raise ValueError("Provide either text query or image")

        faiss.normalize_L2(query_emb)
        scores, indices = self.faiss_index.search(query_emb.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.dataset):
                item = self.dataset.iloc[idx].to_dict()
                item["similarity_score"] = float(score)
                results.append(item)
        return results

    def generate_response(self, query, similar_items):
        #determine gender from query
        gender_hint = ""
        query_lower = query.lower()
        if "men" in query_lower or "male" in query_lower:
            gender_hint = "men's fashion"
        elif "women" in query_lower or "female" in query_lower:
            gender_hint = "women's fashion"
        else:
            gender_hint = "unisex fashion"

        context = ""
        for i, item in enumerate(similar_items, 1):
            context += f"{i}. Description: {item.get('text','No description')}\n"
            if item.get("image"):
                context += "   - Image available\n"
            else:
                context += "   - No image\n"

        prompt = f"""
        You are a professional fashion stylist and personal shopper. 
        The user asked: "{query}" ({gender_hint}).

        Below are the top fashion items retrieved as relevant to the query. 
        Each item includes a description and may have an image:

        {context}

        Your task:
        1. Write a detailed, human-friendly summary of these items.
        2. Highlight key styles, colors, textures, and patterns.
        3. Give practical outfit pairing advice for each item:
        - Tops: suggest pants, skirts, jackets, or accessories.
        - Bottoms: suggest tops, shoes, or layering options.
        - Dresses/One-pieces: suggest shoes, jackets, or accessories.
        4. Recommend suitable occasions (casual, formal, date night, office, etc.).
        5. If images are available, describe notable visual features (like fabric, fit, details).
        6. Include fashion tips like layering, matching colors, and style dos/don'ts.

        Make the summary actionable, detailed, and easy for a person to follow. Use clear examples.
        """
        try:
            response = self.groq_client.chat.completions.create(
                model="gemma-7b-it",
                messages=[
                    {"role": "system", "content": "You are a fashion assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except:
            local_response = "Fashion recommendations based on your query:\n\n"
            for i, item in enumerate(similar_items, 1):
                desc = item.get("text", "No description")
                local_response += f"{i}. {desc}\n"

                # Keyword-based recommendation logic
                if any(word in desc.lower() for word in ["shirt", "top", "t-shirt", "blouse"]):
                    local_response += (
                        "   - Style Tips: Pair with jeans, trousers, or skirts.\n"
                        "   - Layering: Jackets, blazers, or cardigans work well.\n"
                        "   - Accessories: Minimal jewelry, scarves, or belts.\n"
                        "   - Occasions: Casual, office, or evening outings.\n"
                    )
                elif any(word in desc.lower() for word in ["dress", "one-piece"]):
                    local_response += (
                        "   - Style Tips: Pair with heels, flats, or boots.\n"
                        "   - Layering: Jackets or belts to define your waist.\n"
                        "   - Accessories: Earrings, clutch, or subtle jewelry.\n"
                        "   - Occasions: Date nights, casual parties, formal gatherings.\n"
                    )
                elif any(word in desc.lower() for word in ["pants", "jeans", "trousers"]):
                    local_response += (
                        "   - Style Tips: Combine with tucked-in tops or crop tops.\n"
                        "   - Layering: Blazers or coats for sophistication.\n"
                        "   - Accessories: Belts, watches, and matching footwear.\n"
                        "   - Occasions: Office, casual outings, evening events.\n"
                    )
                else:
                    local_response += "   - Style Tips: Mix and match with neutral basics and accessories.\n"
                    local_response += f"   - Similarity Score: {item.get('similarity_score', 0):.3f}\n\n"
            return local_response

    def initialize(self):
        try:
            self.load_dataset()
            self.generate_embeddings()
            self.build_faiss_index()
        except Exception as e:
            print(f"Error during pipeline initialization: {e}")

    def query(self, text_query=None, image_query=None, top_k=5):
        if any(x is None for x in [self.clip_model, self.dataset, self.embeddings, self.faiss_index]):
            self.initialize()
        results = self.search_similar(query=text_query, image=image_query, top_k=top_k)
        desc = text_query if text_query else "uploaded image"
        response = self.generate_response(desc, results)
        return {"query": desc, "similar_items": results, "response": response}

#streamlit App
def run_app():
    st.title("Fashion RAG AI Search")
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = FashionRAGPipeline()
        with st.spinner("Initializing pipeline... This may take a moment."):
            try:
                st.session_state.pipeline.initialize()
            except Exception as e:
                st.error(f"Pipeline initialization failed: {e}")
                return # Stop execution if pipeline fails

    pipeline = st.session_state.pipeline
    tab1, tab2 = st.tabs(["Text Search", "Image Search"])

    #text Search
    with tab1:
        text_query = st.text_input("Enter a fashion query (eg: 'black dress for evening')", key="text_query_input")
        top_k = st.slider("Number of results", 1, 10, 5, key="text_top_k")

        if st.button("Search Text", key="search_text_btn"):
            if text_query:
                with st.spinner("Searching and generating AI summary..."):
                    result = pipeline.query(text_query=text_query, top_k=top_k)
                    # Display results directly within this tab
                    display_results(result)
            else:
                st.warning("Please enter a search query.")

    #image Search
    with tab2:
        uploaded_file = st.file_uploader(
            "Upload an image to find similar fashion items",
            type=["jpg", "jpeg", "png", "webp"],
            key="image_upload_input"
        )
        top_k_img = st.slider("Number of results", 1, 10, 5, key="image_top_k")

        if uploaded_file is not None:
            #displaying the uploaded image immediately
            st.image(uploaded_file, caption="Uploaded Image.", width=200)

            if st.button("Search Image", key="search_image_btn"):
                with st.spinner("Searching and generating AI summary..."):
                    image = Image.open(uploaded_file)
                    result = pipeline.query(image_query=image, top_k=top_k_img)
                    display_results(result)

def display_results(result):
    st.subheader("AI Response")
    
    response_text = result["response"]
    if "unable to generate" in response_text.lower():
        response_text = "Here are some fashion items similar to your query."
    st.write(response_text)

    st.subheader(f"Top {len(result['similar_items'])} Similar Items")
    for item in result["similar_items"]:
        st.write(item.get("text","No description"))

        img_data = item.get("image")
        if img_data:
            try:
                if isinstance(img_data, dict) and "bytes" in img_data:
                    img = Image.open(io.BytesIO(img_data["bytes"]))
                else:  # treat as URL
                    img = Image.open(requests.get(img_data, stream=True).raw)
                st.image(img, width=200)
            except:
                st.write("Image not available")
        else:
            st.write("Image not available")

        st.write(f"Similarity Score: {item.get('similarity_score',0):.3f}")
        st.markdown("---")

#function for CLI
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Text query")
    parser.add_argument("--app", action="store_true", help="Run Streamlit app")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    if args.app:
        run_app()
        return

    if args.query:
        try:
            pipeline = FashionRAGPipeline()
            pipeline.initialize()
        except Exception as e:
            print("Pipeline initialization failed:", e)
            return

        result = pipeline.query(text_query=args.query, top_k=args.top_k)
        print("Query:", result["query"])
        print("Response:", result["response"])
        print("Similar Items:")
        for i, item in enumerate(result["similar_items"],1):
            print(f"{i}. {item.get('text','No description')[:100]}... Score: {item.get('similarity_score',0):.3f}")
    else:
        print("Provide --query or --app")
        
#entry point
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Text query")
    parser.add_argument("--top_k", type=int, default=5)
    args, unknown = parser.parse_known_args()
    
    if args.query:
        #CLI mode
        pipeline = FashionRAGPipeline()
        pipeline.initialize()
        result = pipeline.query(text_query=args.query, top_k=args.top_k)
        print("Query:", result["query"])
        print("Response:", result["response"])
        print("Similar Items:")
        for i, item in enumerate(result["similar_items"], 1):
            print(f"{i}. {item.get('text','No description')[:100]}... Score: {item.get('similarity_score',0):.3f}")
    else:
        run_app()
