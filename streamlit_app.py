import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
from langchain_community.chat_models import ChatPerplexity
import PIL.Image
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Configure API keys and models
@st.cache_resource
def initialize_apis():
    try:
        # Initialize OpenAI
        openai_client = OpenAI(
            api_key="...................."
        )
        
        # Initialize Gemini
        genai.configure(api_key=".............................")
        gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        
        # Initialize Perplexity
        perplexity = ChatPerplexity(
            model="llama-3.1-sonar-small-128k-online",
            api_key="...................................."
        )
        
        return openai_client, gemini_model, perplexity
    except Exception as e:
        st.error(f"Error initializing APIs: {str(e)}")
        return None, None, None

class MedicineAnalyzer:
    def __init__(self, openai_client, gemini_model, perplexity):
        self.df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.openai_client = openai_client
        self.gemini_flash = gemini_model
        self.perplexity = perplexity
        
    def load_data(self, file) -> bool:
        try:
            self.df = pd.read_csv(file)
            self.df['search_text'] = self.df.apply(
                lambda x: f"{str(x['name'])} {str(x['indication'])} {str(x['side_effect'])}", 
                axis=1
            )
            
            self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['search_text'])
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False

    def get_natural_alternatives(self, medicine_name: str) -> Dict[str, Any]:
        if not self.perplexity:
            return {"error": "Perplexity client not initialized"}
        
        try:
            query = f"""For the medicine {medicine_name}, please provide:
            1. If this is an antibiotic, what are the natural alternatives?
            2. List evidence-based natural remedies and lifestyle changes that could help
            3. Provide scientific research supporting these alternatives
            4. Important precautions when using natural alternatives
            5. When medical attention and antibiotics are absolutely necessary
            
            Format the response with clear headings and evidence-based information."""
            
            response = self.perplexity.invoke(query)
            return {"status": "success", "alternatives": response}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def analyze_side_effects_detailed(self, medicines: List[str]) -> Dict[str, Any]:
        if not self.perplexity:
            return {"error": "Perplexity client not initialized"}
        
        try:
            medicines_str = ", ".join(medicines)
            query = f"""Analyze these medicines in detail: {medicines_str}
            
            Provide:
            1. Common and severe side effects for each
            2. Known drug interactions between them (if multiple)
            3. Long-term usage risks
            4. Signs that require immediate medical attention
            5. Safer alternatives if available
            6. Natural remedies that could complement or replace these medicines
            
            Focus on evidence-based information and potential risks."""
            
            response = self.perplexity.invoke(query)
            return {"status": "success", "analysis": response}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def search_medicines(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.df is None or self.tfidf_vectorizer is None or not query:
            return []
        
        try:
            # Split query into individual medicines if multiple are provided
            medicines = [m.strip() for m in query.split(',')]
            
            # Get side effects analysis for all medicines
            side_effects_analysis = self.analyze_side_effects_detailed(medicines)
            
            # Process each medicine
            all_results = []
            for medicine in medicines:
                query_vec = self.tfidf_vectorizer.transform([medicine])
                similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
                top_indices = similarities.argsort()[-top_k:][::-1]
                
                for idx in top_indices:
                    if similarities[idx] > 0:
                        medicine_data = {}
                        for column in self.df.columns:
                            if column != 'search_text':
                                medicine_data[column] = str(self.df.iloc[idx][column])
                        medicine_data['similarity_score'] = similarities[idx]
                        
                        # Get natural alternatives
                        alternatives = self.get_natural_alternatives(medicine_data['name'])
                        medicine_data['natural_alternatives'] = alternatives
                        
                        all_results.append(medicine_data)
            
            # Add overall analysis to results
            if side_effects_analysis.get('status') == 'success':
                all_results.insert(0, {
                    'name': 'Overall Analysis',
                    'type': 'analysis',
                    'content': side_effects_analysis['analysis']
                })
                
            return all_results
        except Exception as e:
            st.error(f"Error searching medicines: {str(e)}")
            return []

    def analyze_with_openai(self, text: str) -> Dict[str, Any]:
        if not self.openai_client:
            return {'analysis': 'OpenAI client not initialized', 'status': 'error'}
            
        try:
            response = self.openai_client.chat.completions.create(
                model="o1-mini",
                messages=[
                    {
                        "role": "user",
                        "content": f"""Analyze this medicine information and provide:
                        1. Key active ingredients
                        2. Common usage scenarios
                        3. Potential risks and side effects
                        4. Whether it's an antibiotic (if applicable)
                        5. Recommendations for safe use
                        6. Natural alternatives (if applicable)
                        7. When to seek medical attention
                        
                        Information to analyze:
                        {text}"""
                    }
                ]
            )
            return {'analysis': response.choices[0].message.content, 'status': 'success'}
        except Exception as e:
            return {'analysis': f"Error in analysis: {str(e)}", 'status': 'error'}

    def analyze_image_with_gemini(self, image) -> str:
        if not self.gemini_flash:
            return "Gemini model not initialized"
            
        try:
            image = image.convert('RGB')
            image_bytes = PIL.Image.new('RGB', image.size)
            image_bytes.paste(image)
            
            prompt = """Analyze this medical image in detail and extract ALL visible information including:

            1. If it's a prescription:
               - Doctor's name and credentials (if visible)
               - Patient's name (if visible)
               - Date of prescription
               - Prescribed medications and dosages
               - Instructions for use
               - Duration of treatment
               - Any special notes or warnings
               - Clinic/Hospital details if available
               - Registration numbers
               - Phone numbers or contact details
            
            2. If it's a medicine package/product:
               - Product name/brand
               - Generic name/composition
               - Strength/dosage form
               - Batch number and expiry date
               - Manufacturer details
               - Storage instructions
               - Warning labels
               - Usage instructions
               - Any visible barcodes or QR codes
               - Natural alternatives if applicable
            
            3. Additional visible elements:
               - Regulatory markings
               - Safety symbols
               - Contact information
               - Website or helpline numbers
               - Any other text visible in the image
            
            Please provide ALL text exactly as written on the image."""
            
            response = self.gemini_flash.generate_content(
                [prompt, image_bytes],
                generation_config={
                    "temperature": 0.1,
                    "top_p": 0.1,
                    "max_output_tokens": 2048,
                },
                safety_settings={
                    "HARM_CATEGORY_HARASSMENT": "block_none",
                    "HARM_CATEGORY_HATE_SPEECH": "block_none",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "block_none",
                    "HARM_CATEGORY_DANGEROUS_CONTENT": "block_none"
                }
            )
            
            if response.text:
                safety_analysis = self.gemini_flash.generate_content(
                    f"""Based on the extracted information:
                    {response.text}
                    
                    Please provide:
                    1. Medicine Classification:
                       - Type of medication (OTC/Prescription)
                       - Therapeutic category
                       - Known drug class
                       - Natural alternatives if applicable
                    
                    2. Safety Considerations:
                       - Standard precautions
                       - Common drug interactions
                       - Storage requirements
                       - Important warnings
                    
                    3. Usage Context:
                       - Typical treatment duration
                       - Common usage scenarios
                       - Administration guidelines
                       - Monitoring requirements"""
                )
                
                combined_response = f"""
                📋 Extracted Information:
                {response.text}
                
                🔍 Additional Analysis:
                {safety_analysis.text}
                """
                return combined_response
            return "No text could be extracted from the image"
            
        except Exception as e:
            return f"Error analyzing image: {str(e)}"

    def get_realtime_info(self, medicine_name: str) -> str:
        if not self.perplexity:
            return "Perplexity client not initialized"
            
        try:
            query = f"""Provide recent information about {medicine_name} focusing on:
            1. Common usage patterns
            2. Known interactions
            3. Latest guidelines
            4. Recent studies or updates
            5. Natural alternatives
            6. Evidence-based complementary treatments
            7. Lifestyle modifications that can help"""
            
            return self.perplexity.invoke(query)
        except Exception as e:
            return f"Could not fetch real-time information. Please check standard drug references."

def main():
    st.set_page_config(
        page_title="Medicine Safety Analyzer",
        page_icon="💊",
        layout="wide"
    )

    st.markdown("""
        <style>
        .main-header {
            color: #2e54a6;
            text-align: center;
            padding: 1rem 0;
        }
        .alert-box {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .warning {
            background-color: #fff3cd;
            border: 1px solid #ffeeba;
        }
        .success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
        }
        .natural-alternatives {
            background-color: #e8f5e9;
            border: 1px solid #c8e6c9;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .stButton > button {
            background-color: #2e54a6;
            color: white;
        }
        .stButton > button:hover {
            background-color: #1e3c7b;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='main-header'>PAMR-Antibiotic Safety Analysis Agent</h1>", unsafe_allow_html=True)

    openai_client, gemini_model, perplexity = initialize_apis()

    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = MedicineAnalyzer(openai_client, gemini_model, perplexity)

    with st.sidebar:
        st.header("📊 Database Management")
        uploaded_file = st.file_uploader("Upload Medicine Database (CSV)", type=['csv'], key="db_uploader")
        if uploaded_file is not None:
            if st.session_state.analyzer.load_data(uploaded_file):
                st.success("✅ Database loaded successfully!")

    tab1, tab2 = st.tabs(["💊 Medicine Search", "📷 Image Analysis"])

    with tab1:
        st.header("Medicine Analysis")
        medicine_text = st.text_area(
            "Enter medicine name(s) (separate multiple medicines with commas):",
            height=100,
            placeholder="Enter medicine name(s)...",
            key="medicine_input"
        )
        
        col1, col2 = st.columns([3,1])
        with col1:
            analyze_button = st.button("🔍 Analyze Medicine", use_container_width=True, key="analyze_med_button")
        
        if analyze_button and medicine_text:
            if st.session_state.analyzer.df is None:
                st.warning("⚠️ Please upload a medicine database first!")
                return

            with st.spinner("🔄 Analyzing..."):
                search_results = st.session_state.analyzer.search_medicines(medicine_text)
                analysis = st.session_state.analyzer.analyze_with_openai(medicine_text)

                if search_results:
                    st.markdown("### 📊 Analysis Results")
                    
                    # Display overall analysis first if available
                    overall_analysis = next((result for result in search_results if result.get('type') == 'analysis'), None)
                    if overall_analysis:
                        st.markdown("#### 🔍 Overall Analysis & Interactions")
                        st.write(overall_analysis['content'])
                        st.markdown("---")
                        st.markdown("### 📋 Individual Medicine Details")
                    for idx, result in enumerate(search_results):
                        if result.get('type') != 'analysis':
                            with st.expander(f"📌 {result.get('name', 'Unknown')} - Match: {result.get('similarity_score', 0):.2f}"):
                                # Basic Information Section
                                st.markdown("#### 📋 Basic Information")
                                st.write("🏢 **Manufacturer:**", result.get('manufacturer', 'N/A'))
                                st.write("💊 **Usage:**", result.get('indication', 'N/A'))
                                
                                # Natural Alternatives Section
                                st.markdown("#### 🌿 Natural Alternatives")
                                alternatives = result.get('natural_alternatives', {})
                                if alternatives.get('status') == 'success':
                                    st.markdown('<div class="natural-alternatives">', unsafe_allow_html=True)
                                    st.write(alternatives['alternatives'])
                                    st.markdown('</div>', unsafe_allow_html=True)
                                else:
                                    st.write("Natural alternatives information not available")
                                
                                # Safety Information Section
                                st.markdown("#### ⚠️ Safety Information")
                                st.markdown("**Contraindications:**")
                                st.write(result.get('contraindication', 'N/A'))
                                st.markdown("**Side Effects:**")
                                st.write(result.get('side_effect', 'N/A'))
                                
                                # Dosage Information
                                st.markdown("#### 💉 Dosage Information")
                                st.write("👤 **Adult Dose:**", result.get('adult_dose', 'N/A'))
                                st.write("👶 **Child Dose:**", result.get('child_dose', 'N/A'))

                                # Real-time Updates Button with unique key
                                button_key = f"search_btn_{idx}_{result.get('name', '').replace(' ', '_')}"
                                if st.button("🔄 Get Latest Info", key=button_key):
                                    with st.spinner("Fetching latest information..."):
                                        info = st.session_state.analyzer.get_realtime_info(result.get('name', ''))
                                        st.info(info)

    with tab2:
        st.header("📷 Prescription & Medicine Image Analysis")
        uploaded_image = st.file_uploader("Upload prescription or medicine image", type=['png', 'jpg', 'jpeg'], key="image_uploader")
        
        if uploaded_image:
            image = PIL.Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("🔍 Analyze Image", key="analyze_image_button", use_container_width=True):
                with st.spinner("🔄 Analyzing image..."):
                    extracted_text = st.session_state.analyzer.analyze_image_with_gemini(image)
                    st.markdown("### 📋 Extracted Information")
                    st.write(extracted_text)
                    
                    if extracted_text and not str(extracted_text).startswith("Error"):
                        analysis = st.session_state.analyzer.analyze_with_openai(extracted_text)
                        if analysis and analysis.get('status') == 'success':
                            st.markdown("### 🔍 Detailed Analysis")
                            st.write(analysis['analysis'])

                        if st.session_state.analyzer.df is not None:
                            search_results = st.session_state.analyzer.search_medicines(extracted_text)
                            if search_results:
                                st.markdown("### 📊 Database Matches")
                                for img_idx, result in enumerate(search_results):
                                    if result.get('type') != 'analysis':
                                        with st.expander(f"📌 {result.get('name', 'Unknown')}"):
                                            # Basic Information
                                            st.markdown("#### 📋 Basic Information")
                                            st.write(f"🏢 **Manufacturer:** {result.get('manufacturer', 'N/A')}")
                                            st.write(f"💊 **Usage:** {result.get('indication', 'N/A')}")

                                            # Natural Alternatives Section
                                            st.markdown("#### 🌿 Natural Alternatives")
                                            alternatives = result.get('natural_alternatives', {})
                                            if alternatives.get('status') == 'success':
                                                st.markdown('<div class="natural-alternatives">', unsafe_allow_html=True)
                                                st.write(alternatives['alternatives'])
                                                st.markdown('</div>', unsafe_allow_html=True)
                                            else:
                                                st.write("Natural alternatives information not available")

                                            # Safety Information
                                            st.markdown("#### ⚠️ Safety Information")
                                            st.markdown("**Contraindications:**")
                                            st.write(result.get('contraindication', 'N/A'))
                                            st.markdown("**Side Effects:**")
                                            st.write(result.get('side_effect', 'N/A'))

                                            # Dosage Information
                                            st.markdown("#### 💉 Dosage Information")
                                            st.write("👤 **Adult Dose:**", result.get('adult_dose', 'N/A'))
                                            st.write("👶 **Child Dose:**", result.get('child_dose', 'N/A'))

                                            # Get Real-time Updates with unique key
                                            img_button_key = f"img_btn_{img_idx}_{result.get('name', '').replace(' ', '_')}"
                                            if st.button("🔄 Get Real-time Info", key=img_button_key):
                                                with st.spinner("Fetching latest information..."):
                                                    real_time_info = st.session_state.analyzer.get_realtime_info(result.get('name', ''))
                                                    st.info(real_time_info)

                                            # Display warnings if available
                                            if result.get('warnings'):
                                                st.markdown("#### 🚨 Important Warnings")
                                                st.warning(result.get('warnings'))
                    else:
                        st.error("❌ No text could be extracted from the image. Please ensure the image is clear and contains readable text.")

if __name__ == "__main__":
    main()
