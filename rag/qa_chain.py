from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

def answer_question(vectorstore, question: str, chat_history: list) -> str:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the question based on the context. "
         "For math use $...$ for inline and $$...$$ for display. If not in context, say so."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    chain = (
        {"context": retriever, "question": RunnablePassthrough(), "chat_history": lambda x: chat_history}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))
    return answer