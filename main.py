from graph import app

def run_rag(question: str):
    print("\n" + "="*50)
    print(f"Question: {question}")
    print("="*50 + "\n")

    inputs = {"question": question, "retries": 0}
    final_generation = ""

    for output in app.stream(inputs):
        for node_name, node_output in output.items():
            print(f"\n--- Output from node: {node_name} ---")
            if "generation" in node_output:
                final_generation = node_output["generation"]

    print("\n" + "="*50)
    print("FINAL ANSWER:")
    print("="*50)
    if not final_generation:
        print("I don't have information about this in the provided documents.")
    else:
        print(final_generation)

if __name__ == "__main__":
    print("RAG system ready.")
    while True:
        question = input("\nAsk your question (or type 'exit' to quit): ").strip()
        if question.lower() == "exit":
            break
        if question:
            run_rag(question)
