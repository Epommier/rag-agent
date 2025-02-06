import streamlit as st
import asyncio

from agent import arun_agent, RagState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Page config
st.set_page_config(
    page_title="Document Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS to change focus color
st.markdown("""
<style>
    /* Change focus color of chat input */
    .stChatInput > div:first-child {
        border-color: #0e1117;
    }
    .stChatInput > div:first-child:hover {
        border-color: #555;
    }
    .stChatInput > div:first-child:focus-within {
        border-color: #0077b6 !important;
        box-shadow: 0 0 0 1px #0077b6 !important;
    }
    
    /* Change placeholder text color */
    .stChatInput textarea::placeholder {
        color: #666;
    }
    
    /* Change input text color */
    .stChatInput textarea {
        color: #fff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title(":speech_balloon: Document Assistant")
    st.markdown("""
    This chatbot uses RAG (Retrieval Augmented Generation) to provide
    informed responses based on your document collection.
    """)
    
    # Add any additional controls here if needed
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.container()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question...", key="chat_input"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    status_container = st.status("Thinking...", expanded=True)

    async def process_agent_response():
        async for state in arun_agent(prompt):
            # Handle system updates
            if "updates" in state and len(state["updates"]) > 0:
                for updt in state["updates"]:                   
                    status_container.write(updt.content)

            # Handle AI messages
            if "messages" in state and len(state["messages"]) > 0:
                msg = state["messages"][-1]                
                if isinstance(msg, AIMessage):
                    
                    status_container.update(label="Done thinking!", expanded=False, state="complete")
                    await asyncio.sleep(0.5)
                    status_container.empty()
                    
                    with st.chat_message("assistant"):
                        st.markdown(msg.content)

                    # Add to chat history only when we get the final answer
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": msg.content
                    })

    # Run the async function
    asyncio.run(process_agent_response())
