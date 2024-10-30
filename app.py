import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import PyPDF2
import google.generativeai as genai
import streamlit as st
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables and configure logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-1.5-pro-latest')

class DataProcessor:
    """Handles loading and preprocessing of both marketing and technical data"""
    
    def __init__(self):
        self.marketing_data = None
        self.error_codes = {}
        
    def load_marketing_data(self, file_path: str):
        """Load marketing data from CSV"""
        try:
            # Read the CSV file
            self.marketing_data = pd.read_csv(file_path)
            
            # Convert date using pandas' flexible parser
            self.marketing_data['date'] = pd.to_datetime(
                self.marketing_data['date'],
                format='mixed',  # Allow mixed formats
                dayfirst=True    # Specify that day comes first in the date string
            )
            
            # Sort by date in descending order
            self.marketing_data = self.marketing_data.sort_values('date', ascending=False)
            
            logger.info("Marketing data loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading marketing data: {e}")
            raise

    def load_error_codes(self, file_path: str):  # Changed method name to match the call
        """Load error codes from PDF"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                current_code = None
                current_section = None
                
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    lines = text.split('\n')
                    
                    for line in lines:
                        if line.strip().isdigit():
                            current_code = line.strip()
                            self.error_codes[current_code] = {
                                'meaning': '',
                                'cause': '',
                                'resolution': ''
                            }
                        elif current_code:
                            if 'Meaning' in line:
                                current_section = 'meaning'
                            elif 'Cause' in line:
                                current_section = 'cause'
                            elif 'Resolution' in line:
                                current_section = 'resolution'
                            elif current_section:
                                self.error_codes[current_code][current_section] += line.strip() + ' '
            
            logger.info("Error codes loaded successfully")
            logger.info(f"Loaded {len(self.error_codes)} error codes")
            
        except Exception as e:
            logger.error(f"Error loading error codes: {e}")
            raise

class MarketingAgent:
    """Agent for handling marketing-related queries"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    def process_query(self, query: str) -> str:
        try:
            latest_data = self.data.iloc[0]
            last_7_days = self.data.head(7)
            
            prompt = f"""
            Based on the following marketing data, please answer this query: {query}
            
            Recent marketing metrics:
            Latest date: {latest_data['date'].strftime('%d-%m-%Y')}
            Today's sales: ${latest_data['sales']:,.2f}
            Last 7 days average sales: ${last_7_days['sales'].mean():,.2f}
            Today's ad spend: ${latest_data['total_ad_spend']:,.2f}
            Meta spend: ${latest_data['corp_Meta_SOCIAL_spend']:,.2f}
            Microsoft spend: ${latest_data['corp_Microsoft_SEARCH_CONTENT_spend']:,.2f}
            """
            
            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Error in marketing agent: {e}")
            return "I apologize, but I encountered an error processing your marketing query."

class TechnicalAgent:
    """Agent for handling technical support queries"""
    
    def __init__(self, error_codes: Dict):
        self.error_codes = error_codes
        
    def process_query(self, query: str) -> str:
        try:
            # Check for error codes in query
            for code in self.error_codes:
                if code in query:
                    error_info = self.error_codes[code]
                    prompt = f"""
                    Please explain this error in simple terms:
                    Error Code: {code}
                    Meaning: {error_info['meaning']}
                    Cause: {error_info['cause']}
                    Resolution: {error_info['resolution']}
                    """
                    
                    response = model.generate_content(prompt)
                    return response.text
                    
            return "I couldn't find a specific error code in your query. Please provide the error code you're seeing."
        except Exception as e:
            logger.error(f"Error in technical agent: {e}")
            return "I apologize, but I encountered an error processing your technical query."

class IntentClassifier:
    """Classifies user queries to determine appropriate agent"""
    
    def __init__(self):
        self.marketing_keywords = {'sales', 'revenue', 'advertising', 'spend', 'marketing', 'promotion'}
        self.technical_keywords = {'error', 'code', 'issue', 'problem', 'troubleshoot', 'fix'}
        
    def classify_intent(self, query: str) -> str:
        query_words = set(query.lower().split())
        
        marketing_score = len(query_words.intersection(self.marketing_keywords))
        technical_score = len(query_words.intersection(self.technical_keywords))
        
        if marketing_score > technical_score:
            return "marketing"
        elif technical_score > marketing_score:
            return "technical"
        return "general"

class ChatbotSystem:
    """Main chatbot system coordinating agents and processing queries"""
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.setup_system()
        
    def setup_system(self):
        """Initialize the system components"""
        try:
            # Load data
            self.data_processor.load_marketing_data("telecom.csv")
            self.data_processor.load_error_codes("error_codes.pdf")
            
            # Initialize agents
            self.marketing_agent = MarketingAgent(self.data_processor.marketing_data)
            self.technical_agent = TechnicalAgent(self.data_processor.error_codes)
            self.intent_classifier = IntentClassifier()
            
            logger.info("Chatbot system initialized successfully")
        except Exception as e:
            logger.error(f"Error setting up chatbot system: {e}")
            raise

    def process_query(self, query: str) -> Dict:
        """Process user query and return response"""
        try:
            intent = self.intent_classifier.classify_intent(query)
            
            if intent == "marketing":
                response = self.marketing_agent.process_query(query)
                source = "Marketing Agent"
            elif intent == "technical":
                response = self.technical_agent.process_query(query)
                source = "Technical Agent"
            else:
                response = "I apologize, but I can only assist with marketing and technical support queries."
                source = "General Handler"
            
            return {
                "query": query,
                "intent": intent,
                "response": response,
                "source": source
            }
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "query": query,
                "intent": "error",
                "response": "I apologize, but I encountered an error processing your request.",
                "source": "Error Handler"
            }

def create_ui():
    """Create Streamlit UI"""
    st.title("TeleCorp USA Customer Service")
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        with st.spinner("Initializing system..."):
            st.session_state.chatbot = ChatbotSystem()
            st.session_state.messages = []
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Query input
    if prompt := st.chat_input("How can I help you today?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Processing your query..."):
                result = st.session_state.chatbot.process_query(prompt)
                st.markdown(result["response"])
                st.caption(f"Processed by: {result['source']}")
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"{result['response']}\n\n*Processed by: {result['source']}*"
        })

if __name__ == "__main__":
    create_ui()