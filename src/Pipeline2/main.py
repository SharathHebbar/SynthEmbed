import streamlit as st
from dotenv import load_dotenv
from utils import get_pdf_text, get_vectorstore, get_text_chunks,text_segment,intent_classifier
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import datetime
import os
import json

# Function to process uploaded PDFs and update metadata
def process_pdfs(pdf_docs, json_filename):
    stored_document_paths = []  # store the paths of the documents

    for pdf_doc in pdf_docs:
        # Check if the file with the same name already exists in the metadata
        filename = pdf_doc.name
        existing_data = {}

        if os.path.exists(json_filename):
            with open(json_filename, "r") as existing_file:
                existing_data = json.load(existing_file)

        if filename in (entry.get("name") for entry in existing_data.values()):
            # Update the timestamp for the existing file
            for entry in existing_data.values():
                if entry.get("name") == filename:
                    entry["date"] = datetime.datetime.now().isoformat()
        else:
            # Saving the uploaded file
            save_directory='Docs/documents'
            os.makedirs(save_directory, exist_ok=True)
            document_path = os.path.join(save_directory, filename)
            
            with open(document_path, "wb") as f:
                f.write(pdf_doc.read())
            stored_document_paths.append(document_path)

            # Metadata creation
            file_details = {
                "name": filename,
                "type": pdf_doc.type,
                "size": pdf_doc.getbuffer().nbytes,
                "date": datetime.datetime.now().isoformat(),
            }

            # Append the new data to the existing data
            existing_data[len(existing_data) + 1] = file_details

        # Write the updated dictionary back to the JSON file
        with open(json_filename, "w") as json_file:
            json.dump(existing_data, json_file)

    return stored_document_paths

def text_file_store(text):
    save_directory = "Docs/textFile"
    os.makedirs(save_directory, exist_ok=True)
    text_path = os.path.join(save_directory,'textract-results.txt')
    
    with open(text_path, 'w+') as f:
        f.write(str(text))


# Function to initialize the Streamlit app
def initialize_app():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    # Metadata Directory
    save_directory = "Docs/metadata"

    # Make sure the directory exists; create it if it doesn't
    os.makedirs(save_directory, exist_ok=True)

    # Define the full path to the JSON file
    json_filename = os.path.join(save_directory, "metadata.json")

    return json_filename

# Main function
def main():
    json_filename=initialize_app()

    st.header("Chat with PDFs :books:")
    user_question = st.text_input("Ask a question about your document:")
    with st.sidebar:
        st.subheader("Your Document")
        pdf_docs = st.file_uploader(
            "Upload your PDF here and click on 'Process'", accept_multiple_files=True,
            type=['pdf', 'txt', 'csv', 'xlsx'])

        if st.button("Process"):
            with st.spinner("Processing"):
                process_pdfs(pdf_docs, json_filename)

                raw_text=" "
                for pdf_doc in pdf_docs:
                    file_extension = pdf_doc.name.split(".")[-1]

                    txt_text=" "
                    pdf_text=" " 
                    # handles txt file
                    if file_extension=='txt':
                        txt_text = pdf_doc.read()
                        txt_text = txt_text.decode('utf-8', errors='ignore')
                        # print(txt_text)

                    # handles pdf file file
                    elif file_extension=='pdf':
                        pdf_text =get_pdf_text(pdf_doc)
                        # print(pdf_text)

                    raw_text +=txt_text+pdf_text
                text_file_store(raw_text)

                # Get the text chunks
                # chunks = get_text_chunks(raw_text)
                

                chunks=text_segment(raw_text)
                st.write(chunks)

                # # Create vector store
                # vectorstore = get_vectorstore(chunks)

                # # Create conversation chain
                # st.session_state.conversation = get_conversation_chain(vectorstore)
            

if __name__ == '__main__':
    main()
