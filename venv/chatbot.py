import os
import json
import openai
import streamlit as st
import numpy as np
from dotenv import load_dotenv
import faiss
from sentence_transformers import SentenceTransformer
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the sentence transformer model


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


model = load_embedding_model()

# Initialize ChatGPT
chat = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Load FAQ dataset and create FAISS index


@st.cache_data
def load_faq_data():
    with open("faq_data.json", "r") as f:
        data = json.load(f)

    # Extract questions and create embeddings
    questions = [faq["question"] for faq in data["faqs"]]
    question_embeddings = model.encode(questions)

    # Create FAISS index
    dimension = question_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(question_embeddings.astype('float32'))

    return data, index, questions


faq_data, faiss_index, questions = load_faq_data()

# Initialize session state variables
if 'chat_memory' not in st.session_state:
    st.session_state.chat_memory = ConversationBufferMemory()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'shown_suggestions' not in st.session_state:
    st.session_state.shown_suggestions = set()

# Add AAMU links dictionary
AAMU_LINKS = {
    "admissions": "https://www.aamu.edu/admissions/",
    "financial aid": "https://www.aamu.edu/admissions-aid/financial-aid/",
    "tuition": "https://www.aamu.edu/admissions-aid/tution-fees/",
    "scholarships": "https://www.aamu.edu/admissions-aid/scholarships/",
    "course registration": "https://www.aamu.edu/about/administrative-offices/student-affairs/registration-information.html",
    "student life": "https://www.aamu.edu/campus-life/",
    "academic calendar": "https://www.aamu.edu/academics/academic-calendar/",
    "housing": "https://www.aamu.edu/campus-life/housing-residence-life/",
    "technology support": "https://www.aamu.edu/campus-life/information-technology/",
    "library": "https://www.aamu.edu/academics/library/",
}


def get_relevant_links(question):
    """Find relevant AAMU links based on the question content."""
    relevant_links = []
    question_lower = question.lower()

    for keyword, link in AAMU_LINKS.items():
        if keyword in question_lower:
            relevant_links.append(
                f"<a href='{link}' target='_blank'>AAMU {keyword.title()} Website</a>")

    return relevant_links


# Initialize conversation chain with memory
chat_with_memory = ConversationChain(
    llm=ChatOpenAI(model_name="gpt-4", temperature=0.7),
    memory=st.session_state.chat_memory
)


def get_gpt4_response(question):
    """Uses memory to make responses context-aware."""
    # Add relevant links to the prompt
    relevant_links = get_relevant_links(question)
    links_text = "\n".join(relevant_links) if relevant_links else ""

    # Load conversation history
    conversation_history = st.session_state.chat_memory.load_memory_variables({
    })
    history = conversation_history.get("history", "")

    system_prompt = f"""You are a helpful assistant for Alabama A&M University. Previous conversation:
{history}

You must format ALL responses EXACTLY like this example:

You can complete the FAFSA application using AAMU's school code 001002. The process is entirely online and should be completed as early as possible to ensure maximum aid consideration.

📌 **Step-by-Step Process:** 
1. Create an account at fafsa.gov
2. Complete the FAFSA form
3. Submit required verification documents
4. Monitor your AAMU email for updates

📌 **Next Steps:** Visit the AAMU Financial Aid Website to begin your application.

Important formatting rules:
1. Start with a direct answer paragraph (no label)
2. Use 📌 emoji before each section header
3. Bold section headers with **
4. Number all steps in the process
5. Include relevant links in Next Steps

Available AAMU links for this response:
{links_text}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"Question: {question}\nPlease provide a response following the exact format shown above.")
    ]

    response = chat(messages)

    # Save the conversation in memory
    st.session_state.chat_memory.save_context(
        {"input": question},
        {"output": response.content}
    )

    # Get suggestions based on the response content
    suggested_topics = get_similar_questions(response.content)
    st.session_state.suggested_topics = suggested_topics

    # Add relevant links if not present in response
    if relevant_links and "Next Steps:" in response.content and not any(link in response.content for link in relevant_links):
        response_content = response.content.replace(
            "📌 **Next Steps:**",
            f"📌 **Next Steps:**\n{' '.join(relevant_links)}"
        )
        return response_content

    return response.content


@st.cache_data
def get_embedding(_model, text):
    """Cache embeddings to avoid recomputing."""
    return _model.encode([text])


@st.cache_data
def search_similar_questions(question_embedding, k=3):
    """Cache FAISS search results."""
    return faiss_index.search(question_embedding.astype('float32'), k=k+1)


def clear_chat():
    """Reset all chat-related session state variables."""
    st.session_state.chat_history = []
    st.session_state.chat_memory = ConversationBufferMemory()
    st.session_state.shown_suggestions = set()
    if 'suggested_topics' in st.session_state:
        del st.session_state.suggested_topics


def get_similar_questions(text, k=3):
    """Get similar questions based on text content."""
    # Use cached embedding computation
    text_embedding = get_embedding(model, text)

    # Use cached search
    distances, indices = search_similar_questions(text_embedding, k)

    # Get similar questions, excluding exact matches
    similar_questions = []
    seen_questions = set()

    for idx in indices[0]:
        question = questions[idx]
        if question not in seen_questions:
            similar_questions.append(question)
            seen_questions.add(question)

    return similar_questions[:3]


def find_answer(question):
    # Handle greetings
    if "hello" in question.lower() or "hi" in question.lower():
        overview = """Welcome to the Alabama A&M University chatbot! 
        
AAMU is a historic public university offering a wide range of undergraduate, graduate, and research programs. We're here to help you learn more about our university.

What would you like to know more about?"""
        # Get suggestions for greeting
        st.session_state.suggested_topics = get_similar_questions(overview)
        return overview

    # Check if it's a follow-up question
    conversation_history = st.session_state.chat_memory.load_memory_variables({
    })
    has_context = conversation_history.get("history", "")

    if has_context and not any(keyword in question.lower() for keyword in ["hello", "hi"]):
        # If there's conversation history, use GPT-4 with memory
        response = get_gpt4_response(question)
        return response

    # Otherwise, proceed with FAQ matching
    # Use cached embedding computation
    question_embedding = get_embedding(model, question)

    # Use cached search
    distances, indices = search_similar_questions(question_embedding, k=1)
    closest_question_idx = indices[0][0]
    similarity_score = distances[0][0]

    SIMILARITY_THRESHOLD = 100

    if similarity_score < SIMILARITY_THRESHOLD:
        # Match found in FAQ
        for idx, faq in enumerate(faq_data["faqs"]):
            if idx == closest_question_idx:
                answer = faq["answer"]
                # Format FAQ answer with sections
                first_sentence = answer.split('.')[0] + '.'
                remaining_details = '. '.join(answer.split('.')[1:]).strip()

                # Convert remaining details into numbered steps
                steps = [step.strip()
                         for step in remaining_details.split('.') if step.strip()]
                numbered_steps = '\n'.join(
                    f"{i+1}. {step}" for i, step in enumerate(steps))

                formatted_answer = f"""{first_sentence}

📌 **Step-by-Step Process:** 
{numbered_steps}

📌 **Next Steps:** """

                # Add relevant links
                relevant_links = get_relevant_links(question)
                if relevant_links:
                    formatted_answer += f"{' '.join(relevant_links)}"
                else:
                    formatted_answer += "Contact the appropriate department for more information at admissions@aamu.edu"

                # Get suggestions based on the full answer
                st.session_state.suggested_topics = get_similar_questions(
                    answer)
                return formatted_answer
    else:
        # No good match found, use GPT-4
        return get_gpt4_response(question)


# Initialize important questions
important_questions = [
    "Is Alabama A&M University accredited?",
    "How do I apply to be admitted to Alabama A&M University?",
    "What materials are needed to complete my AAMU admissions application?"
]

# Streamlit UI
st.title("Alabama A&M University FAQ Bot")

# Add Clear Chat button in the sidebar
with st.sidebar:
    if st.button("🗑️ Clear Chat", help="Reset the conversation"):
        clear_chat()
        st.rerun()

# CSS to keep input box fixed at the bottom
st.markdown("""
    <style>
        .fixed-bottom-input {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background: white;
            padding: 20px;
            box-shadow: 0px -2px 10px rgba(0,0,0,0.1);
            z-index: 100;
        }
        
        /* Add padding to prevent content from being hidden behind fixed input */
        .main-content {
            padding-bottom: 100px;
        }
        
        /* Style the input container */
        .stTextInput > div > div > input {
            border-radius: 20px;
        }

        /* Style the important questions */
        .important-questions {
            margin: 20px 0;
            padding: 15px;
            border-radius: 10px;
            background-color: #f0f2f6;
        }

        .important-questions h4 {
            margin-bottom: 10px;
            color: #1d66dc;
        }

        .clear-chat {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 101;
        }
    </style>
""", unsafe_allow_html=True)

# Create a container for the main content
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Display important questions with styling
st.markdown("""
        <h5>Frequently Asked Questions:</h5>
""", unsafe_allow_html=True)

# Display important questions as buttons
for question in important_questions:
    if st.button(question, key=f"important_{question}"):
        answer = find_answer(question)
        st.session_state.chat_history.extend([
            ("user", question),
            ("bot", answer)
        ])

# Add a divider between sections
st.markdown("---")

# Display chat history and suggestions
for message_idx, (role, message) in enumerate(st.session_state.chat_history):
    if role == "user":
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                <div style="background-color: #1d66dc; color: white; padding: 10px; border-radius: 10px; max-width: 70%;">
                    <b>{message}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
                <div style="background-color: #e3e6e9; color: black; padding: 10px; border-radius: 10px; max-width: 70%;">
                    {message}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Always display suggestions after bot response
        st.markdown("**Related questions you might be interested in:**")
        if hasattr(st.session_state, 'suggested_topics'):
            for suggestion in st.session_state.suggested_topics:
                if st.button(f"📌 {suggestion}", key=f"suggest_{message_idx}_{suggestion}"):
                    answer = find_answer(suggestion)
                    st.session_state.chat_history.extend([
                        ("user", suggestion),
                        ("bot", answer)
                    ])

st.markdown('</div>', unsafe_allow_html=True)  # Close main-content div

# Handle form submission


def handle_input():
    if st.session_state.user_input:
        answer = find_answer(st.session_state.user_input)
        st.session_state.chat_history.extend([
            ("user", st.session_state.user_input),
            ("bot", answer)
        ])
        # Clear input after sending
        st.session_state.user_input = ""


# Fixed input box at the bottom
st.markdown('<div class="fixed-bottom-input">', unsafe_allow_html=True)

col1, col2 = st.columns([6, 1])

with col1:
    st.text_input(
        "Type your question here:",
        key="user_input",
        on_change=handle_input,
        label_visibility="collapsed"
    )

with col2:
    st.button(
        "📤",
        help="Send message",
        on_click=handle_input
    )

st.markdown('</div>', unsafe_allow_html=True)  # Close the fixed div
