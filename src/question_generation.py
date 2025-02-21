import os
import json
import re
from tqdm import tqdm

from LLM import complete_text_openai

MODEL = "gpt-4o"

def strip_html_tags(text: str) -> str:
    """
    Remove HTML tags from the given text.
    """
    clean = re.compile(r'<[^>]+>')
    return re.sub(clean, '', text)

def count_lines(filename):
    with open(filename, "r") as f:
        return sum(1 for _ in f)

def extract_json(response):
    match = re.search(r'```json(.*?)```', response, re.DOTALL)
    if match:
        code_content = match.group(1).strip()
        return code_content
    else:
        return response

def safe_load_json(response):
    output = extract_json(response)
    if not output:
        return []
    try:
        distractors = json.loads(output)
    except Exception as e:
        print(e)
        print("Error parsing distractors from scratch:", output)
        distractors = []
    return distractors


def generate_distractors_from_scratch(formatted_question: str, formatted_answer: str):
    """
    Generate five plausible distractors (incorrect answer options) for a multiple-choice question,
    without any additional context. The distractors should imitate the length and style of the correct answer.
    """
    prompt = (
        "You are provided with a boardgame rules question and its correct answer.\n\n"
        f"Question:\n{formatted_question}\n\n"
        f"Answer:\n{formatted_answer}\n\n"
        "Task: Generate five plausible distractors (incorrect answer options) for a multiple-choice question. "
        "The distractors should imitate "
        "the length and style of the correct answer.\n\n"
        "Output the distractors as a JSON list, for example:\n"
        '["Distractor 1", "Distractor 2", "Distractor 3", "Distractor 4", "Distractor 5"]\n'
	"No additional text outside of the JSON."
    )
    
    output = complete_text_openai(prompt, model=MODEL)
    distractors = safe_load_json(output) 
    return distractors

def generate_distractors_from_rulebook(formatted_question: str, formatted_answer: str, rulebook_text: str):
    """
    Generate five plausible distractors (incorrect answer options) for a multiple-choice question that are
    grounded in the provided rulebook text. The distractors should imitate the length and style of the correct answer.
    """
    prompt = (
        "You are provided with a boardgame rules question, its correct answer, and the rulebook text.\n\n"
        f"Question:\n{formatted_question}\n\n"
        f"Answer:\n{formatted_answer}\n\n"
        f"Rulebook Text:\n{rulebook_text}\n\n"
        "Task: Generate five plausible distractors (incorrect answer options) for a multiple-choice question that are "
        "grounded in the rulebook text. The distractors should imitate the length and style of the correct answer.\n\n"
        "Output the distractors as a JSON list, for example:\n"
        '["Distractor 1", "Distractor 2", "Distractor 3", "Distractor 4", "Distractor 5"]\n'
	"No additional text outside of the JSON."
    )
    
    output = complete_text_openai(prompt, model=MODEL)
    distractors = safe_load_json(output)    
    return distractors

def generate_distractors_from_forum(formatted_question: str, formatted_answer: str, full_content: list):
    """
    Generate five plausible distractors (incorrect answer options) for a multiple-choice question that are
    grounded in the online forum discussion. The discussion is provided in the full_content field, where each
    item is a post. HTML tags in the post content will be removed before generating distractors.
    This function is only called if there are at least three posts in the discussion.
    The distractors should imitate the length and style of the correct answer.
    """
    # Concatenate cleaned forum posts into one discussion text.
    forum_posts = []
    for post in full_content:
        content = post.get("content", "")
        cleaned_content = strip_html_tags(content)
        forum_posts.append(cleaned_content)
    forum_text = "\n".join(forum_posts)
    
    prompt = (
        "You are provided with a boardgame rules question, its correct answer, and an online forum discussion "
        "related to the question."
        f"Question:\n{formatted_question}\n\n"
        f"Answer:\n{formatted_answer}\n\n"
        f"Forum Discussion:\n{forum_text}\n\n"
        "Task: Generate five plausible distractors (incorrect answer options) for a multiple-choice question based on the "
        "discussion content. The distractors should imitate the length and style of the correct answer.\n\n"
        "Output the distractors as a JSON list, for example:\n"
        '["Distractor 1", "Distractor 2", "Distractor 3", "Distractor 4", "Distractor 5"]\n'
	"No additional text outside of the JSON."
    )
    
    output = complete_text_openai(prompt, model=MODEL)
    distractors = safe_load_json(output)    
    return distractors

def process_examples(examples_path: str, rulebook_text: str, output_path: str):
    """
    Process each QA pair from the examples file and generate:
      1. The multiple-choice question (using the original formatted question).
      2. The correct answer (using the original formatted answer).
      3. Five distractors generated from scratch.
      4. Five distractors grounded in the rulebook text.
      5. Optionally, five distractors grounded in online forum discussion if there are at least three posts.
    The final combined result is printed in JSON format.
    """
    processed_examples = []
    total_lines = count_lines(examples_path)
    with open(examples_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Processing questions", total=total_lines):
            example = json.loads(line)
            formatted_question = example["formatted_question"]
            formatted_answer = example["formatted_answer"]

            # Use the original formatted question and answer.
            mcq_question = f"This question is about a boardgame called Pax Renaissance. {formatted_question}"
            correct_answer = formatted_answer

            # Generate distractors from scratch.
            distractors_scratch = generate_distractors_from_scratch(mcq_question, correct_answer)

            # Generate distractors grounded in the rulebook.
            distractors_rulebook = generate_distractors_from_rulebook(mcq_question, correct_answer, rulebook_text)

            # Generate distractors based on forum discussion, if applicable.
            distractors_forum = []
            full_content = example.get("full_content", [])
            if len(full_content) >= 3:
                distractors_forum = generate_distractors_from_forum(mcq_question, correct_answer, full_content)

            # Combine the data into one JSON structure.
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
            processed_examples.append(final_output)

    with open(output_path, 'w') as writer:
        for example in processed_examples:
            writer.write(json.dumps(example)+'\n')
    with open(output_path.replace("jsonl", "json"), 'w') as writer:
        json.dump(processed_examples, writer, indent=2)
    print(f"saved to {output_path}")


def load_rulebook(rulebook_path: str) -> str:
    """
    Load and return the rulebook text from the specified file.
    """
    with open(rulebook_path, "r", encoding="utf-8") as f:
        return f.read()

def main():
    rulebook_path = "rulebook.txt"
    examples_path = "examples.jsonl"
    output_path = "mcq.jsonl"

    rulebook_text = load_rulebook(rulebook_path)
    process_examples(examples_path, rulebook_text, output_path)

if __name__ == "__main__":
    main()

