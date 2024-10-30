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

# Configuration
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro-latest')

class MarketingMetrics:
    """Data model for marketing metrics"""
    def __init__(self, data: pd.Series):
        self.date = data['date']
        self.sales = data['sales']
        self.sales_from_finance = data['sales_from_finance']
        self.total_ad_spend = data['total_ad_spend']
        
        # Calculate platform spends
        self.google_spend = self._calculate_google_spend(data)
        self.meta_spend = data['corp_Meta_SOCIAL_spend']
        self.microsoft_spend = self._calculate_microsoft_spend(data)
        self.local_spend = self._calculate_local_spend(data)
        
        # Financial metrics
        self.stock_market_index = data['stock_market_index']
        self.dollar_to_pound = data['dollar_to_pound']
        self.interest_rates = data['interest_rates']
        
    def _calculate_google_spend(self, data: pd.Series) -> float:
        google_cols = [col for col in data.index if 'corp_Google' in col]
        return sum(data[col] for col in google_cols)
        
    def _calculate_microsoft_spend(self, data: pd.Series) -> float:
        microsoft_cols = [col for col in data.index if 'corp_Microsoft' in col]
        return sum(data[col] for col in microsoft_cols)
        
    def _calculate_local_spend(self, data: pd.Series) -> float:
        local_cols = [col for col in data.index if 'local_' in col]
        return sum(data[col] for col in local_cols)

class DataProcessor:
    """Handles loading and preprocessing of marketing and technical data"""
    
    def __init__(self):
        self.marketing_data = None
        self.error_codes = {}
        
    def load_marketing_data(self, file_path: str):
        """Load and preprocess marketing data"""
        try:
            self.marketing_data = pd.read_csv(file_path)
            
            # Convert date column
            self.marketing_data['date'] = pd.to_datetime(
                self.marketing_data['date'],
                format='%Y-%m-%d'
            )
            
            # Sort by date in descending order
            self.marketing_data = self.marketing_data.sort_values('date', ascending=False)
            
            # Calculate derived metrics
            self._calculate_platform_metrics()
            self._calculate_performance_metrics()
            
            logger.info("Marketing data loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading marketing data: {e}")
            raise
            
    def _calculate_platform_metrics(self):
        """Calculate platform-specific spending metrics"""
        # Google corporate metrics
        google_cols = [col for col in self.marketing_data.columns 
                      if 'corp_Google' in col]
        self.marketing_data['total_google_corp_spend'] = \
            self.marketing_data[google_cols].sum(axis=1)
            
        # Microsoft metrics
        microsoft_cols = [col for col in self.marketing_data.columns 
                         if 'corp_Microsoft' in col]
        self.marketing_data['total_microsoft_spend'] = \
            self.marketing_data[microsoft_cols].sum(axis=1)
            
        # Local advertising metrics
        local_cols = [col for col in self.marketing_data.columns 
                     if 'local_' in col]
        self.marketing_data['total_local_spend'] = \
            self.marketing_data[local_cols].sum(axis=1)
            
    def _calculate_performance_metrics(self):
        """Calculate ROI and efficiency metrics"""
        self.marketing_data['roi'] = (
            self.marketing_data['sales'] / 
            self.marketing_data['total_ad_spend']
        ).round(3)
        
        self.marketing_data['efficiency_ratio'] = (
            self.marketing_data['sales_from_finance'] / 
            self.marketing_data['total_ad_spend']
        ).round(3)

    def load_error_codes(self, file_path: str):
        """Load and parse error codes from PDF"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                current_code = None
                current_section = None
                
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    lines = text.split('\n')
                    
                    for line in lines:
                        # Process error code sections
                        if line.strip().isdigit() or line.strip() in ['015D', '1243a']:
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
            metrics = MarketingMetrics(self.data.iloc[0])
            weekly_metrics = self._calculate_period_metrics(7)
            monthly_metrics = self._calculate_period_metrics(30)
            
            prompt = f"""
            Based on the following marketing data, answer this query: {query}
            
            Current Metrics ({metrics.date.strftime('%Y-%m-%d')}):
            Sales: ${metrics.sales:,.2f}
            Total Ad Spend: ${metrics.total_ad_spend:,.2f}
            
            Platform Spending:
            - Google: ${metrics.google_spend:,.2f}
            - Meta: ${metrics.meta_spend:,.2f}
            - Microsoft: ${metrics.microsoft_spend:,.2f}
            - Local Advertising: ${metrics.local_spend:,.2f}
            
            Weekly Averages:
            - Sales: ${weekly_metrics['avg_sales']:,.2f}
            - Ad Spend: ${weekly_metrics['avg_spend']:,.2f}
            - ROI: {weekly_metrics['avg_roi']:.3f}
            
            Monthly Metrics:
            - Total Sales: ${monthly_metrics['total_sales']:,.2f}
            - Total Ad Spend: ${monthly_metrics['total_spend']:,.2f}
            - Average ROI: {monthly_metrics['avg_roi']:.3f}
            
            Market Indicators:
            - Stock Market Index: {metrics.stock_market_index:,.2f}
            - Dollar to Pound Rate: {metrics.dollar_to_pound:.3f}
            - Interest Rates: {metrics.interest_rates}%
            """
            
            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Error in marketing agent: {e}")
            return "I encountered an error processing your marketing query."
            
    def _calculate_period_metrics(self, days: int) -> Dict:
        period_data = self.data.head(days)
        return {
            'avg_sales': period_data['sales'].mean(),
            'total_sales': period_data['sales'].sum(),
            'avg_spend': period_data['total_ad_spend'].mean(),
            'total_spend': period_data['total_ad_spend'].sum(),
            'avg_roi': period_data['roi'].mean()
        }

class TechnicalAgent:
    """Agent for handling technical support queries"""
    
    def __init__(self, error_codes: Dict):
        self.error_codes = error_codes
        
    def process_query(self, query: str) -> str:
        try:
            # Extract error codes from query
            codes = self._extract_error_codes(query)
            
            if codes:
                responses = []
                for code in codes:
                    error_info = self.error_codes.get(code)
                    if error_info:
                        prompt = f"""
                        Explain this error in simple terms:
                        Error Code: {code}
                        Meaning: {error_info['meaning']}
                        Cause: {error_info['cause']}
                        Resolution: {error_info['resolution']}
                        
                        Please provide a clear, step-by-step explanation.
                        """
                        response = model.generate_content(prompt)
                        responses.append(response.text)
                
                return "\n\n".join(responses)
            
            return "Please provide a specific error code for me to help you troubleshoot."
            
        except Exception as e:
            logger.error(f"Error in technical agent: {e}")
            return "I encountered an error processing your technical query."
            
    def _extract_error_codes(self, query: str) -> List[str]:
        """Extract error codes from query"""
        codes = []
        for code in self.error_codes.keys():
            if code in query:
                codes.append(code)
        return codes

class IntentClassifier:
    """Classifies user queries to determine appropriate agent"""
    
    def __init__(self):
        # Marketing Keywords based on telecom.csv columns
        self.marketing_keywords = {
            # Sales and Revenue
            'sales', 'sales_from_finance', 'revenue', 'earnings',
            
            # Corporate Advertising Platforms
            'corp_google', 'corp_meta', 'corp_microsoft', 'corp_horizon', 'corp_impact',
            'discovery', 'display', 'performance_max', 'search', 'shopping', 'video',
            'social', 'audience', 'affiliate',
            
            # Specific Ad Types
            'corp_google_discovery', 'corp_google_display', 'corp_google_performance_max',
            'corp_google_search', 'corp_google_shopping', 'corp_google_video',
            'corp_horizon_video', 'corp_meta_social', 'corp_microsoft_audience',
            'corp_microsoft_search', 'corp_microsoft_shopping',
            
            # Local Advertising
            'local_google', 'local_meta', 'local_simpli_fi',
            'local_google_display', 'local_google_local',
            'local_google_performance_max', 'local_google_search',
            'local_meta_social', 'local_simpli_fi_geo_optimized',
            
            # Financial Metrics
            'total_ad_spend', 'stock_market_index', 'dollar_to_pound', 'interest_rates',
            'spend', 'budget', 'cost', 'investment',
            
            # Time-based Terms
            'today', 'yesterday', 'last week', 'weekly', 'daily',
            'trend', 'historical', 'performance'
        }
        
        # Technical Keywords based on Error-Codes.pdf
        self.technical_keywords = {
            # Error Codes
            '0 of 0 tuners', '002', 'partial signal loss',
            'channel signal loss', 'programming not authorized',
            
            # Error Related
            'error', 'code', 'issue', 'problem', 'troubleshoot', 'fix',
            'failure', 'malfunction', 'fail state',
            
            # Equipment
            'receiver', 'dish', 'antenna', 'cable', 'tuner',
            'multi-dish switch', 'lnbf', 'smart card', 'component',
            'hard drive', 'power cord', 'front panel reset',
            
            # Signal Related
            'signal', 'moca', 'coax', 'satellite', 'connection',
            'connectivity', 'signal meter', 'signal path',
            'inclement weather', 'misaligned', 'damaged',
            
            # Actions
            'reset', 'reboot', 'check', 'verify', 'examine',
            'install', 'activate', 'authorize', 'check-switch test',
            
            # Status
            'status', 'authorization', 'programming',
            'channel', 'service', 'account', 'activation'
        }
        
    def classify_intent(self, query: str) -> str:
        """
        Classify query intent with enhanced logic and weighting
        """
        query_words = set(query.lower().split())
        
        # Calculate weighted scores
        marketing_score = sum(2 if word in self.marketing_keywords else 0 
                            for word in query_words)
        technical_score = sum(2 if word in self.technical_keywords else 0 
                            for word in query_words)
        
        # Add context-based scoring
        if any(code in query.lower() for code in ['002', '0 of 0', 'tuner']):
            technical_score += 3
            
        if any(platform in query.lower() for platform in 
               ['google', 'meta', 'microsoft', 'horizon', 'impact', 'simpli.fi']):
            marketing_score += 3
            
        # Check for specific ad spend patterns
        if 'spend' in query.lower() and any(platform in query.lower() 
                                          for platform in ['google', 'meta', 'microsoft']):
            marketing_score += 2
            
        # Check for specific error patterns
        if 'signal' in query.lower() and 'loss' in query.lower():
            technical_score += 2
            
        # Determine intent based on scores
        if marketing_score > technical_score:
            return "marketing"
        elif technical_score > marketing_score:
            return "technical"
        
        # If scores are equal, look for specific patterns
        if any(error_indicator in query.lower() 
               for error_indicator in ['error', 'issue', 'problem', 'fix']):
            return "technical"
        
        if any(marketing_indicator in query.lower() 
               for marketing_indicator in ['sales', 'advertising', 'spend']):
            return "marketing"
            
        return "general"
    
    def get_confidence_score(self, query: str) -> float:
        """
        Calculate confidence score for the classification
        """
        query_words = set(query.lower().split())
        marketing_matches = len(query_words.intersection(self.marketing_keywords))
        technical_matches = len(query_words.intersection(self.technical_keywords))
        total_words = len(query_words)
        
        if total_words == 0:
            return 0.0
            
        max_matches = max(marketing_matches, technical_matches)
        confidence = max_matches / total_words
        return min(confidence, 1.0)  # Cap at 1.0

class TeleCorpChatbot:
    """Main chatbot system"""
    
    def __init__(self, marketing_data_path: str, error_codes_path: str):
        self.data_processor = DataProcessor()
        self.setup_system(marketing_data_path, error_codes_path)
        
    def setup_system(self, marketing_data_path: str, error_codes_path: str):
        try:
            # Load data
            self.data_processor.load_marketing_data(marketing_data_path)
            self.data_processor.load_error_codes(error_codes_path)
            
            # Initialize components
            self.marketing_agent = MarketingAgent(self.data_processor.marketing_data)
            self.technical_agent = TechnicalAgent(self.data_processor.error_codes)
            self.intent_classifier = IntentClassifier()
            
            logger.info("Chatbot system initialized successfully")
        except Exception as e:
            logger.error(f"Error setting up chatbot: {e}")
            raise
            
    def process_query(self, query: str) -> Dict:
        try:
            intent = self.intent_classifier.classify_intent(query)
            
            if intent == "marketing":
                response = self.marketing_agent.process_query(query)
                source = "Marketing Analytics System"
            elif intent == "technical":
                response = self.technical_agent.process_query(query)
                source = "Technical Support System"
            else:
                response = "I can only assist with marketing and technical support queries."
                source = "General Handler"
                
            return {
                "query": query,
                "intent": intent,
                "response": response,
                "source": source
            }
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return {
                "query": query,
                "intent": "error",
                "response": "An error occurred processing your request.",
                "source": "Error Handler"
            }

def create_ui():
    """Create Streamlit UI"""
    st.title("TeleCorp USA Customer Service")
    
    if 'chatbot' not in st.session_state:
        with st.spinner("Initializing system..."):
            st.session_state.chatbot = TeleCorpChatbot("telecom.csv", "error_codes.pdf")
            st.session_state.messages = []
    
    # Display chat history
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

if __name__ == "__main__":
    create_ui()
