def build_context(results) -> str:
    contexts = []

    for i, (result, score) in enumerate(results, start=1):
        label = f"[Source: {result.payload['source']} — page {result.payload['page']}]"
        if result.payload.get("type") == "diagram_caption":
            label += " (AI-generated description of a diagram/schematic on this page)"

        context = f"{label}\n{result.payload['content']}"

        contexts.append(f"[context {i}]\n{context}")

    return "\n\n".join(contexts)
