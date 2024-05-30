# Ask_your_pdf

Objective:  Project aims to facilitate users in querying information from multiple PDF documents uploaded by user.



Python: Utilized as the primary programming language.
Streamlit: Employed for building the user interface, making it interactive and user-friendly.
Google GEMINI-pro: Integrated for its natural language processing capabilities, allowing users to input questions in a conversational manner.
Langchain: Presumably used for text summarization or language processing tasks.
Functionality:

User Input: Users provide questions they want to ask about the content of the PDF documents.
PDF Upload: Users upload PDF documents containing the relevant information.
Processing: The uploaded PDFs are processed through Langchain, to extract and understand the textual content.
Querying: The Google GEMINI-PRO   is employed to interpret and understand user questions, generating relevant queries from them.
Matching: The system matches the generated queries with the extracted information from the PDFs.
Summarization: Langchain may be used to summarize the relevant information retrieved from the PDFs, providing concise answers to the user queries.
Display: The summarized information is displayed to the user through the Streamlit interface.
