# System Architecture Documentation

## Overview
The TeleCorp USA Multi-Agent Chatbot System implements a modular, multi-agent architecture designed for scalability and maintainability.

## Core Components

### 1. Data Layer
- **DataProcessor**
  - Handles data loading and preprocessing
  - Manages CSV and PDF data sources
  - Implements data cleaning and validation

### 2. Agent Layer
- **MarketingAgent**
  - Processes marketing and sales queries
  - Accesses marketing metrics and KPIs
  - Generates data-driven responses

- **TechnicalAgent**
  - Handles technical support queries
  - Processes error code documentation
  - Provides troubleshooting guidance

### 3. Intent Classification Layer
- **IntentClassifier**
  - Determines query type
  - Routes queries to appropriate agents
  - Handles edge cases

### 4. Interface Layer
- **Streamlit UI**
  - Chat interface
  - Session state management
  - Response rendering

## Data Flow
1. User Input → Intent Classification
2. Intent Classification → Agent Selection
3. Agent → Data Access
4. Data Processing → Response Generation
5. Response → UI Rendering

## System Dependencies
```mermaid
graph TD
    A[User Interface] --> B[Intent Classifier]
    B --> C[Agent Router]
    C --> D[Marketing Agent]
    C --> E[Technical Agent]
    D --> F[Marketing Data]
    E --> G[Error Codes]
