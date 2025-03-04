import json
import random
import string
import os
import asyncio
import concurrent.futures

from src.llm_infer import llm_infer

random.seed(12345)

# Create a semaphore allowing up to 20 concurrent calls
MAX_CONCURRENT = 40
semaphore = asyncio.Semaphore(MAX_CONCURRENT)

async def evaluate_single(question, choices, correct_idx, distractor_type, url, context_data):
    """
    Asynchronously evaluate a single (question, distractor) combination under a given context.
    Uses a semaphore to limit concurrency to MAX_CONCURRENT tasks at once.

    Returns:
      (
        predicted_idx (int),
        correct_flag (1 or 0),
        prediction_explanation (str),
        metadata (dict)  # identifying info for storing the result
      )
    """
    # Build the request string for llm_infer
    context_str = []
    
    # Insert reference material
    for source in context_data.get("sources_data", []):
        label = source["label"]
        text = source["content"]
        context_str.append(f"[Reference Material Description]: {label}\n")
        context_str.append(f"[Reference Material Content]: {text}\n\n")

    context_str.append(f"QUESTION: {question}\n")
    
    # Multiple-choice options
    import string
    context_str.append("MULTIPLE-CHOICE OPTIONS:")
    for label_char, choice_text in zip(string.ascii_lowercase, choices):
        context_str.append(f"  {label_char}) {choice_text}")

    if len(context_data.get("sources_data", [])) > 0:
        context_str.append(
            "INSTRUCTIONS: provide a json response that contains a 'predicted_index' field "
            "for the NUMERIC index of the multiple-choice question that you deem correct. "
            "The json should also contain an 'explanation' field that justifies the answer. You must provide quotes/citations to the reference material provided.\n"
        )
    else:
        context_str.append(
            "INSTRUCTIONS: provide a json response that contains a 'predicted_index' field "
            "for the NUMERIC index of the multiple-choice question that you deem correct. "
            "The json should also contain an 'explanation' field that justifies the answer.\n"
        )

    prompt = "\n".join(context_str)

    # Acquire the semaphore before making the LLM call
    async with semaphore:
        loop = asyncio.get_running_loop()
        # We'll offload the synchronous llm_infer call to a ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor() as pool:
            raw_output_list = await loop.run_in_executor(
                pool, lambda: llm_infer([prompt], use_json=True)
            )

    # raw_output_list should have one item (the LLM response)
    output_str = raw_output_list[0] if raw_output_list else ""

    try:
        output = json.loads(output_str)
    except json.JSONDecodeError:
        # If the model didn't return valid JSON, handle gracefully
        output = {
            "predicted_index": "0",
            "explanation": f"Model did not return valid JSON. Raw response: {output_str}"
        }

    predicted_idx = int(output.get("predicted_index", 0))
    correct_flag = 1 if predicted_idx == correct_idx else 0
    prediction_explanation = output.get("explanation", "No explanation provided.")

    metadata = {
        "question": question,
        "url": url,
        "choices": choices,
        "correct_index": correct_idx,
        "predicted_index": predicted_idx,
        "correct_flag": correct_flag,
        "context_name": context_data["context_name"],
        "distractor_type": distractor_type,
        "prediction_explanation": prediction_explanation,
    }

    return predicted_idx, correct_flag, prediction_explanation, metadata


async def main_async():
    # 1. Load the context configurations
    context_file_path = "context_configs/pax_ren_2e.json"
    with open(context_file_path, "r", encoding="utf-8") as cf:
        context_configs = json.load(cf)

    # 2. Load the question data
    questions_file_path = "data/mcq_100.json"
    with open(questions_file_path, "r", encoding="utf-8") as qf:
        questions_data = json.load(qf)

    # We'll store tasks here
    tasks = []

    # Specify the required distractor types
    required_distractor_types = ["from_scratch", "from_rulebook", "from_forum"]

    # 3. Create tasks for each (context, question, distractor_type)
    for context_obj in context_configs:
        context_name = context_obj["name"]
        sources = context_obj["sources"]
        sources_data = []

        # Load source files
        for src in sources:
            for label, file_path in src.items():
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                else:
                    content = f"[File Not Found: {file_path}]"
                sources_data.append({
                    "label": label,
                    "content": content
                })

        context_data = {
            "context_name": context_name,
            "sources_data": sources_data
        }

        # For each question + distractor type
        for example in questions_data:
            distractors_dict = example.get("distractors", {})
            
            # Skip this question if it doesn't have ALL required distractor types
            if not all(dt in distractors_dict for dt in required_distractor_types):
                continue

            question = example["multiple_choice_question"]
            correct_answer = example["correct_answer"]
            url = example["url"]

            # Now iterate through each required distractor type
            for distractor_type in required_distractor_types:
                distractor_list = distractors_dict[distractor_type]
                
                all_choices = [correct_answer] + distractor_list
                random.shuffle(all_choices)
                correct_idx = all_choices.index(correct_answer)

                # Create an async task
                task = asyncio.create_task(
                    evaluate_single(
                        question=question,
                        choices=all_choices,
                        correct_idx=correct_idx,
                        distractor_type=distractor_type,
                        url=url,
                        context_data=context_data
                    )
                )
                tasks.append(task)

    # 4. Run all tasks concurrently with a limit of MAX_CONCURRENT at once
    all_results = await asyncio.gather(*tasks)

    # 5. Store results in results_by_context
    results_by_context = {}
    for (_, _, _, metadata) in all_results:
        ctx = metadata["context_name"]
        dt = metadata["distractor_type"]

        if ctx not in results_by_context:
            results_by_context[ctx] = {
                "from_scratch": [],
                "from_rulebook": [],
                "from_forum": []
            }
        if dt not in results_by_context[ctx]:
            results_by_context[ctx][dt] = []

        results_by_context[ctx][dt].append(metadata)

    # 6. Save the results
    detailed_results_file = "100_evaluation_results.json"
    with open(detailed_results_file, "w", encoding="utf-8") as outfile:
        json.dump(results_by_context, outfile, indent=2, ensure_ascii=False)

    print(f"\nDetailed results have been written to {detailed_results_file}.")

    # 7. Generate an accuracy report
    accuracy_report = {}
    for ctx_name, dt_dict in results_by_context.items():
        accuracy_report[ctx_name] = {}
        for distractor_type, results_list in dt_dict.items():
            if len(results_list) == 0:
                accuracy_report[ctx_name][distractor_type] = {
                    "num_examples": 0,
                    "accuracy": 0.0
                }
            else:
                correct_flags = [r["correct_flag"] for r in results_list]
                accuracy = sum(correct_flags) / len(correct_flags)
                accuracy_report[ctx_name][distractor_type] = {
                    "num_examples": len(correct_flags),
                    "accuracy": accuracy
                }

    accuracy_file = "100_accuracy_report.json"
    with open(accuracy_file, "w", encoding="utf-8") as rep:
        json.dump(accuracy_report, rep, indent=2, ensure_ascii=False)

    print(f"Accuracy report has been written to {accuracy_file}.")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
