from langgraph.graph import END, StateGraph
from graph_state import GraphState
from nodes import (
    retrieve,
    grade_documents,
    generate,
    transform_query,
    hallucination_grader,
    answer_grader
)

def decide_to_generate(state: GraphState):
    print("--- ROUTER: DECIDE TO GENERATE ---")

    if state.get("retries",0)>=3:
        print("Decision: Max retries reached, cannot answer")
        return "end"

    if state["web_search"] == "Yes":
        print("Decision: rewrite query")
        return "transform_query"
    print("Decision: generate answer")
    return "generate"

def grade_generation(state: GraphState):
    print("--- ROUTER: GRADE GENERATION ---")
    documents = state["documents"]
    generation = state["generation"]
    question = state["question"]

    hallucination_score = hallucination_grader.invoke({
        "documents": documents,
        "generation": generation
    })

    if hallucination_score.binary_score == "yes":
        print("Decision: answer is grounded, checking if it resolves question")
        answer_score = answer_grader.invoke({
            "question": question,
            "generation": generation
        })
        if answer_score.binary_score == "yes":
            print("Decision: answer is useful, returning to user")
            return "useful"
        print("Decision: answer does not resolve question, rewriting query")
        return "not useful"

    print("Decision: hallucination detected, regenerating")
    return "not supported"

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

workflow.set_entry_point("retrieve")

workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
        "end":END
    }
)

workflow.add_edge("transform_query", "retrieve")

workflow.add_conditional_edges(
    "generate",
    grade_generation,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    }
)

app = workflow.compile()
print("Graph compiled successfully.")
