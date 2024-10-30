# TeleCorp USA Multi-Agent Chatbot System

## Overview
An intelligent customer service chatbot that uses multiple specialized agents to handle marketing and technical support queries. The system leverages Google's Gemini API for natural language processing and implements a RAG (Retrieval-Augmented Generation) approach for accurate responses.

## Features
- Multi-agent architecture with specialized handlers
- Intent-based query routing
- Real-time data processing
- Interactive chat interface
- Marketing and technical support capabilities
- Comprehensive logging system

## Prerequisites
- Python 3.8+
- Google API Key (Gemini)
- Required data files:
  - telecom.csv (marketing data)
  - error_codes.pdf (technical documentation)

## Installation

1. Clone the repository:

git clone https://github.com/iswarit/telecom_usa_multi_agent_llm.git
cd telecom_usa_multi_agent_llm

## Install Requirements

pip install -r requirements.txt

## Start the application (offline)

1. python -m streamlit run app.py
2. Access the web interface at http://localhost:8501
3. Enter queries in the chat interface

## Example Queries

1. Marketing
   "What were our total sales for last week?"
   "How much did we spend on Meta advertising yesterday?"
   "What's our current interest rate?"
   
3. Technical
   "What does error code 002 mean?"
   "How do I fix signal loss?"
   "My receiver shows error 002, what should I do?"

Example usage Screenshot also attached with this repo.![app_page-0004](https://github.com/user-attachments/assets/013bc808-ffc2-416b-be8e-1017f7dd5435)
![app_page-0003](https://github.com/user-attachments/assets/ca43556e-8b92-4559-9467-f5459f1ff92f)
![app_page-0002](https://github.com/user-attachments/assets/703b5805-1404-4f58-b997-07db293d1ac8)
![app_page-0001](https://github.com/user-attachments/assets/a3ccc872-8e56-419a-956c-ae8b482ff371)
![app_page-0005](https://github.com/user-attachments/assets/d0c309ee-30f5-40b7-9ba0-fa764b14c8f9)
