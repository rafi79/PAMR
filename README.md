# PAMR-Antibiotic Safety Analysis Agent 🏥

An intelligent Agent that analyzes medicines, provides natural alternatives, and helps make informed healthcare decisions using advanced AI and machine learning techniques.

## 🌟 Features

### 1. Intelligent Medicine Analysis
- Multi-medicine interaction checking
- Comprehensive side effects analysis
- Real-time safety information updates
- Evidence-based natural alternatives suggestions

### 2. Advanced Search Capabilities
- TF-IDF vectorization for precise medicine matching
- Cosine similarity for finding related medicines
- Multi-dimensional comparison of:
  - Medicine names
  - Indications
  - Side effects
  - Usage patterns

### 3. Image Analysis
- Prescription text extraction
- Medicine package analysis
- Automatic information categorization
- Safety warnings detection

### 4. Natural Alternatives
- Evidence-based natural remedy suggestions
- Scientific research citations
- Safety precautions for natural alternatives
- Integration with medical databases

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI/ML**:
  - OpenAI (medicine analysis)
  - Google Gemini (image processing)
  - Perplexity (real-time information)
  - scikit-learn (TF-IDF & similarity)
- **Data Processing**: 
  - Pandas
  - NumPy
- **Image Processing**: PIL
- **Natural Language Processing**:
  - TF-IDF Vectorization
  - Cosine Similarity

## 📋 Requirements

```bash
Python 3.8+
streamlit
pandas
numpy
scikit-learn
python-dotenv
openai
google-generativeai
langchain-community
Pillow
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/medicine-safety-analyzer.git
cd medicine-safety-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with your API keys:
```env
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_gemini_key
PERPLEXITY_API_KEY=your_perplexity_key
```

4. Prepare your medicine database:
Create a CSV file with the following columns:
- name
- manufacturer
- indication
- side_effect
- contraindication
- adult_dose
- child_dose
- warnings (optional)

5. Run the application:
```bash
streamlit run app.py
```

## 💡 Usage

### Medicine Analysis
1. Upload your medicine database CSV file
2. Enter medicine name(s) in the search box
3. View comprehensive analysis including:
   - Side effects
   - Interactions
   - Natural alternatives
   - Safety information

### Image Analysis
1. Upload a prescription or medicine package image
2. Get automatic text extraction and analysis
3. View matches from your medicine database
4. Access real-time safety information

## 🔍 Technical Implementation

### TF-IDF and Cosine Similarity

The system uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert medicine descriptions into numerical vectors:

```python
def load_data(self, file):
    self.df['search_text'] = self.df.apply(
        lambda x: f"{str(x['name'])} {str(x['indication'])} {str(x['side_effect'])}", 
        axis=1
    )
    
    self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['search_text'])
```

Cosine similarity is then used to find related medicines:

```python
similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
top_indices = similarities.argsort()[-top_k:][::-1]
```

### Medicine Search Algorithm

The search process follows these steps:

1. Text Preprocessing:
   - Combine medicine name, indication, and side effects
   - Remove stop words and special characters
   - Convert to lowercase for standardization

2. Vector Creation:
   - Convert text to TF-IDF vectors
   - Capture importance of terms in medicine descriptions

3. Similarity Calculation:
   - Compute cosine similarity between query and database
   - Rank results by similarity score
   - Filter based on threshold values

4. Results Processing:
   - Retrieve top matches
   - Fetch additional information
   - Get natural alternatives

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for medicine analysis capabilities
- Google for Gemini Vision API
- Perplexity for real-time information
- The medical community for valuable feedback
