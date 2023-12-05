import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

os.environ["OPENAI_API_KEY"] = "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"

def main():
    st.title("Audit Report Insights: The Key to Making Informed Financial Decisions")

    with open(r"D:\accumulate\20220202_alphabet_10K.txt", encoding='iso-8859-1') as f:
        state_of_the_union = f.read()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(state_of_the_union)

    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))]).as_retriever()
    stock_name = st.text_input("Enter Stock:", "")  # New text box for entering stock name
    st.write("Entered stock:", stock_name)

    selected_question = st.selectbox("Select your question", [
        "What is the scope of the audit? What areas of the company's financial statements were audited?",
        "Were any significant accounting policies or estimates changed during the year? If so, how did the auditor evaluate these changes?",
        "What were the auditor's overall findings and conclusions? Were there any material misstatements or weaknesses in internal controls?",
        "Did the auditor identify any fraud or other irregularities? If so, how did they respond to these findings?",
        "What were the auditor's recommendations for improving the company's financial reporting processes?",
        "Did the auditor provide an opinion on the company's financial statements? If so, what was the opinion?",
        "Were there any disagreements between the auditor and management? If so, how were these disagreements resolved?",
        "Did the auditor provide any additional assurance services beyond the standard audit? If so, what were the results of these services?",
        "What was the auditor's fee for the audit and related services?",
        "What is the auditor's track record in terms of identifying financial reporting issues or weaknesses in internal controls at other companies?"
    ])

    st.write("Selected question:", selected_question)

    docs = docsearch.get_relevant_documents(selected_question)

    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
    answers = chain.run(input_documents=docs, question=selected_question)
    st.success('Answer: {}'.format(answers))

if __name__ == '__main__':
    main()
