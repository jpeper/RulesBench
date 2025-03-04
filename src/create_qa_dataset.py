import argparse
from llm_infer import llm_infer
import json

def process_json_to_qa(json_data, max_examples, filter_rules_questions):
    qa_dataset = []
    
    for thread_id, thread_data in list(json_data.items())[:max_examples]:
        og_posts = thread_data["posts"]
        posts = json.dumps(og_posts, indent=2)
        
        llm_prompt = f"""You are provided with a series of forum posts about a game rules question. I want to clean this up into a QA example.
I basically want to parse the example, obtain the original question, find a 'ground-truth' answer that can be inferred from the discussion (note there might be multiple proposed, but use the context to infer which one is correct. It could be the last, or the one that people clearly agree on etc...). Note, make sure the processed question and answer fully reflect the original question -- I don't want to lose any details or comprehensivity in the process. I also want to track which original responses the ultimate question and answer are found in. Also, let's generate some metadata about the example, e.g. for if the example contains a rules question (True/False) and also if the question was answered during the thread (True/False) (some examples might not be relevant, and some might be unanswered).
Concretely, output a json file with the following fields: 'formatted_question', 'formatted_answer', 'question_citation_indices', 'answer_citation_indices', 'contains_rules_question', 'is_answered'. Here is the post: {posts}"""
        
        try:
            output = json.loads(llm_infer([llm_prompt], model='gpt-4o', use_json=True)[0])
            if filter_rules_questions and (not output.get('contains_rules_question', False)) and (not output.get('is_answered', False)):
                continue
            
            output['raw_question'] = "\n".join([og_posts[idx]['content'] for idx in output['question_citation_indices']])
            output['raw_answer'] = "\n".join([og_posts[idx]['content'] for idx in output['answer_citation_indices']])
            output['full_content'] = og_posts
            output['url'] = thread_data['url']
            output['topic'] = thread_data['subject']
            qa_dataset.append(output)
        except Exception as e:
            print(f"Error processing thread {thread_id}: {e}")

    return qa_dataset

def main():
    parser = argparse.ArgumentParser(description="Process JSON file to QA dataset.")
    parser.add_argument("input_file", type=str, help="Path to input JSON file")
    parser.add_argument("output_file", type=str, help="Path to output file")
    parser.add_argument("--max_examples", type=int, default=10, help="Maximum number of examples to process")
    parser.add_argument("--output_format", choices=["json", "jsonl"], default="jsonl", help="Output format: json or jsonl")
    parser.add_argument("--filter_rules_questions", action="store_true", help="Only include examples that contain a rules question")
    args = parser.parse_args()
    
    with open(args.input_file, "r", encoding="utf-8") as infile:
        json_data = json.load(infile)
    
    qa_dataset = process_json_to_qa(json_data, args.max_examples, args.filter_rules_questions)
    
    with open(args.output_file, "w", encoding="utf-8") as outfile:
        if args.output_format == "json":
            json.dump(qa_dataset, outfile, indent=4)
        else:
            for entry in qa_dataset:
                outfile.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    main()