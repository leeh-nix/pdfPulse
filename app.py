import streamlit as st
from backend.extract import extract_text_from_pdf, split_text
from backend.pinecone import (
    initialize_pinecone,
    setup_index,
    store_in_pinecone,
    query_pinecone,
)
from backend.generate_response import (
    initialize_gemini,
    generate_embeddings,
    generate_response,
)

from log import logger

# Page Configuration
st.set_page_config(
    page_title="pdfPulse",
    layout="centered",
    page_icon="😈",
)

# Sidebar: API Key Inputs
st.sidebar.subheader(":green[API Configuration]")
GEMINI_API = (
    st.sidebar.text_input(
        ":blue[Gemini API Key]",
        type="password",
        help="Enter your Gemini API key.",
    )
    or st.secrets["GEMINI_API_KEY"]
)
PINECONE_API = (
    st.sidebar.text_input(
        ":violet[Pinecone API Key]",
        type="password",
        help="Enter your Pinecone API key.",
    )
    or st.secrets["PINECONE_API"]
)

st.session_state.index = ""
global queryable


if GEMINI_API and PINECONE_API:
    try:
        initialize_gemini(GEMINI_API)

        # Initialize Pinecone index
        index_options = initialize_pinecone(PINECONE_API)
        INDEX_NAME = st.sidebar.selectbox(
            "Select an Index", index_options + ["Create new index"]
        )
        if INDEX_NAME == "Create new index":
            INDEX_NAME = st.sidebar.text_input(
                "Pinecone Index",
                max_chars=45,
                help="Name must consist of lower case alphanumeric characters or '-'. _Defaults to pdf-pulse_",
            )
            if not INDEX_NAME or 1 > len(INDEX_NAME) > 45:
                st.error("Please enter a valid index name.")
        try:
            st.session_state.index = setup_index(PINECONE_API, INDEX_NAME)
            queryable = True
        except Exception as e:
            queryable = False
            st.error(f"{e}")
        st.sidebar.success("API keys configured successfully!")
    # except (UnauthorizedException or PineconeApiValueError) as e:
    except Exception as e:
        queryable = False
else:
    queryable = False
    st.sidebar.warning("Please configure your API keys and index name.")

st.sidebar.markdown(":red[_Your API Keys are used locally and is NOT saved._]")

st.header("pdfPulse")

uploaded_file = st.file_uploader("Upload a file (PDF only)", type=["pdf"])


# Initialize session state variables

if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

if "file_processed" not in st.session_state:
    st.session_state.file_processed = False

if "chunks_stored" not in st.session_state:
    st.session_state.chunks_stored = False

if "text" not in st.session_state:
    st.session_state.text = ""

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = []


if uploaded_file:
    if not st.session_state.uploaded:
        st.write(f"File '{uploaded_file.name}' uploaded successfully.")
        st.session_state.uploaded = True

        # Extract text from PDF
        if not st.session_state.file_processed:
            st.write("Processing the file...")
            st.session_state.text = extract_text_from_pdf(uploaded_file)
            st.session_state.file_processed = True
            st.success("File processed successfully!")
    else:
        st.info("File has already been uploaded and processed.")

# Chunking and storing embeddings
if st.session_state.file_processed and not st.session_state.chunks_stored:
    # Check if index is initialized
    if st.session_state.index is not None:
        st.session_state.chunks = split_text(st.session_state.text)
        st.session_state.embeddings = generate_embeddings(st.session_state.chunks)

        store_in_pinecone(
            st.session_state.index, st.session_state.chunks, st.session_state.embeddings
        )

        st.session_state.chunks_stored = True
        st.success("Chunks stored in Pinecone successfully!")
    else:
        st.error("Pinecone index not initialized. Please enter API keys.")


# Chat
chat_history = st.container()

if "messages" not in st.session_state:
    st.session_state["messages"] = []


# TODO - provide previous messages too as context to the ai
# TODO - free api has around 1500 word limit for pdf, how to solve it?
# TODO - option to input index name (for deletion as well)


# Query input
if queryable:  # type: ignore
    st.subheader(f"Index: {INDEX_NAME}")  # type:ignore
    if query := st.chat_input("Ask a question..."):
        st.session_state["messages"].append({"role": "user", "content": query})
        with st.spinner("Generating response..."):
            query_embedding = generate_embeddings([query])[0]
            relevant_chunks = query_pinecone(st.session_state.index, query_embedding, top_k=5)  # type: ignore
            context = "\n".join(relevant_chunks)
            ai_response = generate_response(query, context)
        st.session_state["messages"].append(
            {"role": "assistant", "content": ai_response}
        )

        # Refresh chat display
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
else:  # if api keys not entered
    st.error("Please configure API keys correctly.")
# Footer
st.sidebar.markdown(
    """
    [![GitHub](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/leeh-nix/PDF-Chatbot)
    """,
    unsafe_allow_html=True,
)
