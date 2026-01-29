from rag_retriever import RAGRetriever


r = RAGRetriever(
"rag_index/index.faiss",
"rag_index/data.json"
)


results = r.retrieve("What is recursion?")
print(results)