I'll create comprehensive documentation for the AAMU FAQ Chatbot project.

```markdown:faq_chatbot/venv/README.md
# AAMU FAQ Chatbot Documentation

## Overview
The AAMU FAQ Chatbot is an intelligent conversational agent designed to assist users with questions about Alabama A&M University. It combines semantic search, natural language processing, and a user-friendly interface to provide accurate, context-aware responses.

## Features
- 🤖 Natural language understanding
- 🔍 Semantic search for FAQ matching
- 💭 Context-aware responses with conversation memory
- 🔗 Automatic link suggestions to relevant AAMU resources
- 💡 Dynamic topic suggestions
- 🎯 Step-by-step process guidance
- 🔄 Clear chat functionality

## Technical Stack
- **Python 3.7+**
- **Streamlit**: Web interface framework
- **LangChain**: Language model chain operations
- **FAISS**: Vector similarity search
- **Sentence Transformers**: Text embeddings
- **OpenAI GPT-4**: Advanced language model

## Project Structure
```

faq_chatbot/
├── venv/ # Virtual environment
├── data/
│ └── faq_data.json # FAQ database
├── chatbot.py # Main application
├── requirements.txt # Dependencies
└── README.md # Documentation

````

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd faq_chatbot
````

2. Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:

```bash
export OPENAI_API_KEY=your_api_key_here
```

5. Run the application:

```bash
streamlit run chatbot.py
```

## Core Components

### 1. Embedding Model

- Uses `SentenceTransformer('all-MiniLM-L6-v2')` for text embeddings
- Cached loading for improved performance
- Converts questions and responses into vector representations

### 2. FAISS Index

- Efficient similarity search for FAQ matching
- Optimized with caching for faster responses
- Maintains vector database of FAQ questions

### 3. Conversation Memory

- Tracks chat history using `ConversationBufferMemory`
- Enables context-aware responses
- Persists during user session

### 4. Link Management

```python
AAMU_LINKS = {
    "admissions": "https://www.aamu.edu/admissions/",
    "financial aid": "https://www.aamu.edu/admissions-aid/financial-aid/",
    # ... other links ...
}
```

## Response Format

Responses follow a consistent structure:

```
[Clear answer paragraph]

📌 **Step-by-Step Process:**
1. [First step]
2. [Second step]
3. [Third step]

📌 **Next Steps:** [Relevant AAMU links]
```

## Key Functions

### get_gpt4_response(question)

- Generates context-aware responses using GPT-4
- Maintains conversation memory
- Formats responses with steps and links

### find_answer(question)

- Main answer generation pipeline
- Combines FAQ matching and GPT-4 responses
- Handles greeting messages and follow-ups

### get_similar_questions(text, k=3)

- Generates relevant topic suggestions
- Uses semantic similarity for matching
- Returns top 3 related questions

## Performance Optimizations

- `@st.cache_resource`: Model loading
- `@st.cache_data`: FAQ data and embeddings
- Cached FAISS searches
- Optimized embedding computations
- Reduced API calls

## User Interface Features

1. **Chat Interface**

   - Message history display
   - Fixed input box
   - Formatted responses

2. **Navigation**

   - Clear chat button
   - Clickable suggestions
   - Important questions shortcuts

3. **Visual Elements**
   - Emoji indicators
   - Formatted links
   - Step-by-step processes

## Maintenance

### Adding New FAQs

1. Update `faq_data.json`:

```json
{
  "faqs": [
    {
      "question": "New question?",
      "answer": "Detailed answer with steps."
    }
  ]
}
```

2. Restart the application to update embeddings

### Updating Links

1. Modify `AAMU_LINKS` dictionary in `chatbot.py`
2. Links automatically integrate into responses

## Troubleshooting

### Common Issues

1. **Slow Responses**

   - Check internet connection
   - Verify API key status
   - Clear browser cache

2. **Missing Links**

   - Verify AAMU_LINKS dictionary
   - Check keyword matching
   - Ensure HTML rendering is enabled

3. **Format Issues**
   - Verify system prompt formatting
   - Check markdown rendering
   - Review response structure

## Future Enhancements

1. Multi-language support
2. Voice interface integration
3. Custom training data integration
4. Advanced analytics dashboard
5. User feedback system

## Contributing

1. Fork the repository
2. Create feature branch
3. Submit pull request
4. Follow coding standards
5. Add appropriate documentation

## License

[Specify License]

## Contact

[Specify Contact Information]

```

This documentation provides:
- Complete setup instructions
- Technical architecture details
- Usage guidelines
- Maintenance procedures
- Troubleshooting help
- Future development plans

It should help users and developers understand, use, and maintain the chatbot effectively.
```
