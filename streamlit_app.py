import streamlit as st
from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage
from graphs.rag_graph import agentic_rag_graph
import seaborn as sns
import re
import os
import requests
import time
import pandas as pd
import matplotlib.pyplot as plt

def get_key_output(key, value):
    outputs = {
        'agent': lambda: 'Calling agent... ✅\n' if value['messages'][0].response_metadata['finish_reason'] == 'stop' else 'Fetching relevant information...✅\n',
        'retrieve': lambda: f"Looked up: {','.join(re.findall(r'Source File: (.*)', value['messages'][0].content))} ... ✅\n",
        'generate': lambda: "Check hallucinations and generate answer ...✅\n",
        'rewrite': lambda: f"Rewriting user question as: {value['messages'][0].content} \n",
        'translate_answer': lambda: "Translating answer to user's language ...✅ \n"
    }
    return outputs.get(key, lambda: f"Processing at step: {key} ...✅\n")()

def predict(message, history, uploaded_file=None):
    history_langchain_format = [
        HumanMessage(content=msg['content'], additional_kwargs={'image': msg.get('image')})
        if msg['role'] == 'human' else AIMessage(content=msg['content'])
        for msg in history
    ]

    if uploaded_file:
        history_langchain_format.append(HumanMessage(content=message, additional_kwargs={'image': uploaded_file}))
    else:
        history_langchain_format.append(HumanMessage(content=message))

    inputs = {"messages": history_langchain_format}
    partial_message = ""
    final_answer = ''

    for i, output in enumerate(agentic_rag_graph.stream(inputs, config={"recursion_limit": 25}, stream_mode="updates")):
    # for i, output in enumerate({}):   
        for key, value in output.items():
            partial_message += get_key_output(key, value)
            if key == 'translate_answer':
                final_answer = value['messages'][0].content
            yield partial_message
            time.sleep(0.1)

    yield final_answer

def make_call(phone):
    phone = "+91" + phone
    url = "https://api.bolna.dev/call"
    payload = {
        "agent_id": "1402b21f-7ecb-4b18-ad4b-68fe2192fed2",
        "recipient_phone_number": phone
    }
    headers = {
        "Authorization": f"Bearer {os.getenv('BOLNA_API')}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return "Call will be received shortly."
    except requests.exceptions.RequestException as e:
        return f"Failed to make the call: {e}"

def chat_page():
    st.header("Chat with Kimaya - Your Personal Cutomer Manager")
    st.markdown("This is a demo of D2C brand selling grass trimmer and their customer has recently purchased product - STIHL FS120 Brush Cutter. \n Customer can ask any questions related to product usage as well as how to make best use of thier product. \n Features: Understanding input image, local language support etc")
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello Ishwar! Thank you for purchasing the STIHL FS120 Brush Cutter. I'm Kimaya, your customer assistant. I can help you make best use of your product, can you provide me a picture of your lawn? Or if you need any other help regarding the product feel free to ask me!"}
        ]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "human" and "image" in message:
                st.image(message["image"], caption="Uploaded Image", use_column_width=True)
            st.markdown(message["content"])

    # Chat input
    with st.container():
        with st.form("chat_input_form"):
            user_input = st.text_input("Type your message here...")
            uploaded_file = st.file_uploader("Upload an image of your lawn or equipment", type=["jpg", "png", "jpeg"])
            submit_button = st.form_submit_button("Send")

            if submit_button:
                # Add user message to chat history
                st.session_state.messages.append({"role": "human", "content": user_input})
                if uploaded_file:
                    st.session_state.messages[-1]["image"] = uploaded_file

                # Display user message
                with st.chat_message("human"):
                    if uploaded_file:
                        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                    st.markdown(user_input)

                # Generate and display assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response_generator = predict(user_input, st.session_state.messages, uploaded_file)
                        response_placeholder = st.empty()
                        for response in response_generator:
                            response_placeholder.markdown(response)

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

                # Rerun the app to update the chat history
                st.rerun()
    with st.container():
        st.markdown("<h3>Examples</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h4>1. Suggest me suitable attachments and tips to mown my lawn</h4>", unsafe_allow_html=True)
            uploaded_file = "data/images/lawn_image.jpg"
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            if st.button("Ask", key="lawn_example", use_container_width=True):
                user_input = "Suggest me suitable attachments for my brush cutter and tips to mown my lawn"
                # Add user message to chat history
                st.session_state.messages.append({"role": "human", "content": user_input, "image": uploaded_file})
                # Display user message
                with st.chat_message("human"):
                    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                    st.markdown(user_input)
                # Generate and display assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response_generator = predict(user_input, st.session_state.messages, uploaded_file)
                        response_placeholder = st.empty()
                        for response in response_generator:
                            response_placeholder.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                # Rerun the app to update the chat history
                st.rerun()
        with col2:
            st.markdown("<h4>2. How to service my machine?</h4>", unsafe_allow_html=True)
            if st.button("Ask", key="service_machine", use_container_width=True):
                user_input = "How can I service my STIHL FS120 Brush Cutter?"
                # Add user message to chat history
                st.session_state.messages.append({"role": "human", "content": user_input})
                # Display user message
                with st.chat_message("human"):
                    st.markdown(user_input)
                # Generate and display assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response_generator = predict(user_input, st.session_state.messages)
                        response_placeholder = st.empty()
                        for response in response_generator:
                            response_placeholder.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                # Rerun the app to update the chat history
                st.rerun()
        # New example
        with st.container():
            st.markdown("<h4>3. कृपया आवासीय क्षेत्र काटने के लिए घास ट्रिमर के लिए उपयुक्त सहायक उपकरण का सुझाव दें।</h4>", unsafe_allow_html=True)
            if st.button("Ask", key="hindi_example", use_container_width=True):
                user_input = "कृपया आवासीय क्षेत्र काटने के लिए घास ट्रिमर के लिए उपयुक्त सहायक उपकरण का सुझाव दें।"
                # Add user message to chat history
                st.session_state.messages.append({"role": "human", "content": user_input})
                # Display user message
                with st.chat_message("human"):
                    st.markdown(user_input)
                # Generate and display assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response_generator = predict(user_input, st.session_state.messages)
                        response_placeholder = st.empty()
                        for response in response_generator:
                            response_placeholder.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                # Rerun the app to update the chat history
                st.rerun()

def call_page():
    st.header("Product Feedback Call \n -  Sploot (InfoEdge Funded D2C Brand)")
    st.markdown("Demo of Kimaya communicating with the customer to understand their experience about the product - Sploot Dog Food")
    phone_number = st.text_input("Enter your 10-digit phone number", max_chars=10)
    if st.button("Request Call Back", use_container_width=True):
        if len(phone_number) == 10 and phone_number.isdigit():
            result = make_call(phone_number)
            st.success(result)
        else:
            st.error("Please enter a valid 10-digit phone number.")


def analytics_page():
    st.header("📊 Analytics Demo")
    st.markdown("User Engagement Analysis")

    # Sample dummy data
    data = {
        'Date': pd.date_range(start='2022-01-01', periods=10, freq='D'),
        'Users Engaged': [100, 120, 110, 130, 140, 150, 160, 170, 180, 190],
        'Product Enquiries': [20, 25, 18, 30, 28, 35, 40, 45, 50, 55],
        'General Questions': [30, 28, 35, 27, 32, 34, 36, 38, 40, 42],
        'Warranty Claims': [5, 7, 6, 8, 9, 10, 11, 12, 13, 14],
        'Product Page Visits': [200, 220, 210, 230, 240, 250, 260, 270, 280, 290],
        'KPI: Repeat Purchases': [50, 55, 53, 60, 65, 70, 75, 80, 85, 90]
    }

    df = pd.DataFrame(data)

    # Calculate percentage change in KPI Metric
    df['KPI % Change'] = df['KPI: Repeat Purchases'].pct_change().fillna(0) * 100

    # Set up the layout
    col1, col2 = st.columns(2)

    # Line chart for users engaged
    with col1:
        st.subheader("Users Engaged Over Time")
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='Date', y='Users Engaged', data=df, marker='o', errorbar='sd')
        plt.xlabel("Date")
        plt.ylabel("Users Engaged")
        plt.xticks(rotation=45)
        st.pyplot(plt)

    # Bar chart for types of enquiries
    with col2:
        st.subheader("Types of Enquiries")
        enquiry_types = ['Product Enquiries', 'General Questions', 'Warranty Claims']
        enquiry_counts = df[enquiry_types].sum().reset_index()
        enquiry_counts.columns = ['Enquiry Type', 'Count']
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Enquiry Type', y='Count', data=enquiry_counts, palette="viridis", hue='Enquiry Type', dodge=False, legend=False)
        plt.xlabel("Enquiry Type")
        plt.ylabel("Count")
        st.pyplot(plt)

    # Additional complex plot: Heatmap of enquiries over time
    st.subheader("Heatmap of Enquiries Over Time")
    heatmap_data = df.melt(id_vars=['Date'], value_vars=enquiry_types, var_name='Enquiry Type', value_name='Count')
    plt.figure(figsize=(12, 8))
    heatmap_pivot = heatmap_data.pivot(index='Date', columns='Enquiry Type', values='Count')
    sns.heatmap(heatmap_pivot, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={'label': 'Count'})
    plt.gca().yaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
    st.pyplot(plt)

    # Line chart for product page visits
    st.subheader("Product Page Visits Over Time")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Date', y='Product Page Visits', data=df, marker='o', errorbar='sd')
    plt.xlabel("Date")
    plt.ylabel("Product Page Visits")
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # Line chart for KPI Metric and its percentage change
    st.subheader("KPI: Repeat Purchases and % Change Over Time")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    sns.lineplot(x='Date', y='KPI: Repeat Purchases', data=df, marker='o', errorbar='sd', ax=ax1, label='KPI: Repeat Purchases')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("KPI: Repeat Purchases")
    ax1.tick_params(axis='x', rotation=45)

    ax2 = ax1.twinx()
    sns.lineplot(x='Date', y='KPI % Change', data=df, marker='o', errorbar='sd', ax=ax2, color='r', label='% Change')
    ax2.set_ylabel("% Change in KPI: Repeat Purchases")

    fig.tight_layout()
    st.pyplot(fig)

def main():
    st.set_page_config(page_title="Kimaya", layout="wide")

    st.title("Kimaya", anchor=False)

    # Custom CSS for improved UI
    st.markdown("""
    <style>
    .main {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    .stTextInput > div > div > input {
        font-size: 1.2rem;
        padding: 0.5rem;
    }
    .stButton>button {
        font-size: 1.2rem;
        padding: 0.5rem 1rem;
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stRadio > div {
        gap: 1rem;
    }
    .stRadio > div > label {
        font-size: 1.2rem;
    }
    .sidebar .sidebar-content {
        font-size: 1.2rem;
    }
    .sidebar .sidebar-content a {
        text-decoration: none;
        color: #4CAF50;
    }
    .sidebar .sidebar-content a:hover {
        color: #2c3e50;
    }
    .sidebar .sidebar-content .stRadio > div {
        gap: 1rem;
    }
    .sidebar .sidebar-content .stRadio > div > label {
        font-size: 1.2rem;
    }
    .sidebar .sidebar-content h2 {
        font-size: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar for navigation
    st.sidebar.title("🔍 Navigation")
    st.sidebar.markdown("<h2>Select Demo to View</h2>", unsafe_allow_html=True)

    page = st.sidebar.radio("Select Demo to View", ["💬 Chat Demo", "📞 Feedback Call Demo", "📊 Analytics Demo"])

    if page == "💬 Chat Demo":
        chat_demo = st.sidebar.radio("Select a product", ["Brush Cutter Customer"])
        if chat_demo == "Brush Cutter Customer":
            chat_page()
    elif page == "📞 Feedback Call Demo":
        call_page()
    elif page == "📊 Analytics Demo":
        analytics_page()

if __name__ == "__main__":
    main()
