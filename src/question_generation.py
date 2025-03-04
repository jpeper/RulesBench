import os
import json
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Import your llm_infer
from src.llm_infer import llm_infer

def strip_html_tags(text: str) -> str:
    """
    Remove HTML tags from the given text.
    """
    clean = re.compile(r'<[^>]+>')
    return re.sub(clean, '', text)

def count_lines(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def extract_json(response):
    """
    Attempts to extract JSON content from a fenced code block like:
    ```json
    { ... }
    ```
    If not found, returns the full response string.
    """
    match = re.search(r'```json(.*?)```', response, re.DOTALL)
    if match:
        code_content = match.group(1).strip()
        return code_content
    else:
        return response

def safe_load_json(response):
    """
    Extracts JSON text from the response (if present) and parses it.
    Falls back gracefully if parsing fails.
    """
    output = extract_json(response)
    if not output:
        return []
    try:
        distractors = json.loads(output)
    except Exception as e:
        print(e)
        print("Error parsing distractors:", output)
        distractors = []
    return distractors

async def generate_distractors_from_scratch_async(question: str, answer: str, semaphore: asyncio.Semaphore):
    prompt = (
        "You are provided with a boardgame rules question and its correct answer.\n\n"
        f"Question:\n{question}\n\n"
        f"Correct Answer:\n{answer}\n\n"
        "Task: Generate a multiple-choice question including the correct answer and up to five plausible distractors (incorrect answer options)"
        "The distractors should mimic the length and style of the correct answer. Each distractor should have the same number of clauses as the original answer to mitigate spurious length biases.\n\n"
        "Output the multiple-choice question as a JSON list, with the correct answer as the first element. e.g. {'mcq_candidates': <list>}\n"
    )

    async with semaphore:
        loop = asyncio.get_running_loop()
        output_list = await loop.run_in_executor(
            None,  # Use default ThreadPoolExecutor
            lambda: llm_infer([prompt], use_json=True)
        )
    output = json.loads(output_list[0])['mcq_candidates'][1:] if output_list else ""
    return output

async def generate_distractors_from_rulebook_async(question: str, answer: str, rulebook_text: str, semaphore: asyncio.Semaphore):
    prompt = (
        "You are provided with a boardgame rules question, its correct answer, and the rulebook text.\n\n"
        f"Question:\n{question}\n\n"
        f"Correct Answer:\n{answer}\n\n"
        f"Rulebook Text:\n{rulebook_text}\n\n"
        "Task: Generate a multiple-choice question including the correct answer and up to five plausible distractors (incorrect answer options)"
        "The distractors should mimic the length and style of the correct answer. Each distractor should have the same number of clauses as the original answer to mitigate spurious length biases.\n\n"
        "Output the multiple-choice question as a JSON list, with the correct answer as the first element. e.g. {'mcq_candidates': <list>}\n"
    )

    async with semaphore:
        loop = asyncio.get_running_loop()
        output_list = await loop.run_in_executor(
            None,
            lambda: llm_infer([prompt], use_json=True)
        )
    output = json.loads(output_list[0])['mcq_candidates'][1:] if output_list else ""
    return output

async def generate_distractors_from_forum_async(question: str, answer: str, full_content: list, semaphore: asyncio.Semaphore):
    forum_posts = []
    for post in full_content:
        content = post.get("content", "")
        cleaned_content = strip_html_tags(content)
        forum_posts.append(cleaned_content)
    forum_text = "\n".join(forum_posts)

    prompt = (
        "You are provided with a boardgame rules question, its correct answer, and an online forum discussion.\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Forum Discussion:\n{forum_text}\n\n"
        "Task: Generate a multiple-choice question including the correct answer and up to five plausible distractors (incorrect answer options)"
        "The distractors should mimic the length and style of the correct answer. You can use the provided discussion to potentially source distractors. If they exist, try to source them fairly extractively when possible, and augment these with generated ones if there are not enough. Each distractor should have the same number of clauses as the original answer to mitigate spurious length biases.\n\n"
        "Output the multiple-choice question as a JSON list, with the correct answer as the first element. e.g. {'mcq_candidates': <list>}\n"
    )

    async with semaphore:
        loop = asyncio.get_running_loop()
        output_list = await loop.run_in_executor(
            None,
            lambda: llm_infer([prompt], use_json=True)
        )
    output = json.loads(output_list[0])['mcq_candidates'][1:] if output_list else ""
    return output

async def generate_distractors_from_forum_and_rulebook_async(
    question: str,
    answer: str,
    full_content: list,
    rulebook_text: str,
    semaphore: asyncio.Semaphore
):
    """
    Generate five plausible distractors that leverage *both* the forum discussion
    (cleaned of HTML) and the rulebook text as context.
    """
    # Gather forum text
    forum_posts = []
    for post in full_content:
        content = post.get("content", "")
        cleaned_content = strip_html_tags(content)
        forum_posts.append(cleaned_content)
    forum_text = "\n".join(forum_posts)

    prompt = (
        "You are provided with a boardgame rules question, its correct answer, an online forum discussion, "
        "and the official rulebook text. Use both of these sources as context.\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        "Forum Discussion:\n"
        f"{forum_text}\n\n"
        "Rulebook Text:\n"
        f"{rulebook_text}\n\n"
        "Task: Generate a multiple-choice question including the correct answer and up to five plausible distractors (incorrect answer options)"
        "The distractors should mimic the length and style of the correct answer. You can use the provided forum discussion to potentially source distractors. If they exist, try to source them fairly extractively when possible, and augment these with generated distractors (inspired by the rulebook knowledge) if there are not enough. Each distractor should have the same number of clauses as the original answer to mitigate spurious length biases.\n\n"
        "Output the multiple-choice question as a JSON list, with the correct answer as the first element. e.g. {'mcq_candidates': <list>}\n"
    )

    async with semaphore:
        loop = asyncio.get_running_loop()
        output_list = await loop.run_in_executor(
            None,
            lambda: llm_infer([prompt], use_json=True)
        )
    output = json.loads(output_list[0])['mcq_candidates'][1:] if output_list else ""
    return output

async def process_single_example(example, rulebook_text, semaphore):
    formatted_question = example["formatted_question"]
    formatted_answer = example["formatted_answer"]

    mcq_question = f"This question is about a boardgame called Pax Renaissance Second Edition. {formatted_question}"
    correct_answer = formatted_answer

    # Start tasks in parallel (all at once), then await them:
    task_scratch = generate_distractors_from_scratch_async(mcq_question, correct_answer, semaphore)
    task_rulebook = generate_distractors_from_rulebook_async(mcq_question, correct_answer, rulebook_text, semaphore)

    # We'll check if forum content is large enough for forum-based distractors
    full_content = example.get("full_content", [])
    task_forum = None
    task_forum_rulebook = None

    if len(full_content) >= 3:
        task_forum = generate_distractors_from_forum_async(mcq_question, correct_answer, full_content, semaphore)
        task_forum_rulebook = generate_distractors_from_forum_and_rulebook_async(
            mcq_question,
            correct_answer,
            full_content,
            rulebook_text,
            semaphore
        )

    # Gather the results
    distractors_scratch, distractors_rulebook = await asyncio.gather(task_scratch, task_rulebook)
    distractors_forum = []
    distractors_forum_and_rulebook = []

    if task_forum and task_forum_rulebook:
        distractors_forum, distractors_forum_and_rulebook = await asyncio.gather(task_forum, task_forum_rulebook)

    final_output = {
        "multiple_choice_question": mcq_question,
        "correct_answer": correct_answer,
        "distractors": {
            "from_scratch": distractors_scratch,
            "from_rulebook": distractors_rulebook
        },
        "url": example["url"],
    }

    # Only include forum-based distractors if they were generated.
    if distractors_forum:
        final_output["distractors"]["from_forum"] = distractors_forum

    # And likewise for the new type combining both forum + rulebook
    if distractors_forum_and_rulebook:
        final_output["distractors"]["from_forum_and_rulebook"] = distractors_forum_and_rulebook

    return final_output

async def process_examples_async(examples_path: str, rulebook_text: str, output_path: str, max_concurrency=25):
    semaphore = asyncio.Semaphore(max_concurrency)
    processed_examples = []
    total_lines = count_lines(examples_path)

    tasks = []
    with open(examples_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading examples", total=total_lines):
            example = json.loads(line)
            task = process_single_example(example, rulebook_text, semaphore)
            tasks.append(task)

    print(f"Generating distractors with concurrency={max_concurrency}...")
    results = await asyncio.gather(*tasks)
    processed_examples.extend(results)

    # Write out the results
    with open(output_path, 'w', encoding='utf-8') as writer:
        for ex in processed_examples:
            writer.write(json.dumps(ex, ensure_ascii=False) + '\n')

    # Also save as a .json if desired
    with open(output_path.replace("jsonl", "json"), 'w', encoding='utf-8') as writer:
        json.dump(processed_examples, writer, indent=2, ensure_ascii=False)

    print(f"Saved to {output_path}")

def load_rulebook(rulebook_path: str) -> str:
    with open(rulebook_path, "r", encoding="utf-8") as f:
        return f.read()

def main():
    # Adjust these paths as needed
    rulebook_path = "rules_material/pax_ren_2e/paxren_rulebook1.txt"
    examples_path = "data/paxren_100_hot.jsonl"
    output_path = "data/mcq_100.jsonl"

    rulebook_text = load_rulebook(rulebook_path)

    # Run the async event loop
    asyncio.run(process_examples_async(examples_path, rulebook_text, output_path, max_concurrency=25))

if __name__ == "__main__":
    main()
