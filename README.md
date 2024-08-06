# LLM Chatbot News Insights

This Streamlit application allows users to get instant insights from online news articles using a language model chatbot. 
Users can enter their Google API key, provide URLs to news articles, and ask questions about the content of those articles. 
The app processes the articles, creates a vector store using FAISS, and uses the Google Generative AI model to generate answers.

## Features

- Load news articles from provided URLs.
- Split the text into manageable chunks.
- Create a vector store using FAISS.
- Ask questions about the articles and get detailed answers from a language model.

## How to Use

1. Clone the repository or copy the code to your local machine.
2. Install the required dependencies.
3. Run the Streamlit app.

### Prerequisites

- Python 3.7 or higher
- Streamlit
- LangChain
- Google Generative AI SDK
- FAISS

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/llm-chatbot-news-insights.git
   cd llm-chatbot-news-insights```

2. **Install dependencies**
	
	```pip install streamlit langchain google-generative-ai faiss-cpu```
	
### Running the app

1. **Using .py file**
	```streamlit run file_name.py ```
	
2. **Using LocalTunnel to expose the app**
	```!curl ipv4.icanhazip.com ```
    ```!streamlit run app.py &>./logs.txt & npx localtunnel --port 8501 ```

Streamlit app link: https://rag-lllm-newsbot-nxqhbws3bf4laexgqbmkcx.streamlit.app/
	
