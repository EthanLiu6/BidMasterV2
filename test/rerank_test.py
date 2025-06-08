from pymilvus.model.reranker import CrossEncoderRerankFunction

ce_rf = CrossEncoderRerankFunction(
    # model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",  # Specify the model name.
    model_name="../models/thenlper/gte-large-zh",  # Specify the model name.
    device="mps"  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
)

reranked_results = ce_rf(
    query='What event in 1956 marked the official birth of artificial intelligence as a discipline?',
    documents=[
        "In 1950, Alan Turing published his seminal paper, 'Computing Machinery and Intelligence,' proposing the Turing Test as a criterion of intelligence, a foundational concept in the philosophy and development of artificial intelligence.",
        "The Dartmouth Conference in 1956 is considered the birthplace of artificial intelligence as a field; here, John McCarthy and others coined the term 'artificial intelligence' and laid out its basic goals.",
        "In 1951, British mathematician and computer scientist Alan Turing also developed the first program designed to play chess, demonstrating an early example of AI in game strategy.",
        "The invention of the Logic Theorist by Allen Newell, Herbert A. Simon, and Cliff Shaw in 1955 marked the creation of the first true AI program, which was capable of solving logic problems, akin to proving mathematical theorems."
    ],
    top_k=3
)

for result in reranked_results:
    print(f'score: {result.score}')
    print(f'doc_text: {result.text}')
