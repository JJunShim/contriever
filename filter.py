import os
import json
import configparser
import openai
import time

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Tuple
from openai import OpenAI, OpenAIError, RateLimitError

api_key = ""
client = OpenAI(api_key=api_key)

def read_jsonl(filepath: str) -> List[Tuple[int, str]]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return [(i, json.loads(line.strip())["text"]) for i, line in enumerate(f) if line.strip()]

def gpt_classify(text: str, system_prompt: str, max_retries=5) -> bool:
    prompt = f"""
    {system_prompt}

    Now evaluate:
    Input: "{text}"
    Output:
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",  # or "gpt-4" if needed
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            probability = [_.logprobs for _ in response.choices]
            content = " ".join([
                value for key, value in response.choices[
                    probability.index(max(probability))
                ].message if key == "content"
            ])
            return "true" in content.strip().lower()
        except RateLimitError:
            print(f"[재시도 {attempt+1}] Rate limit. 잠시 대기 중...")
            time.sleep(2 ** attempt)
        except OpenAIError as e:
            print(f"[오류] OpenAI API 에러: {e}")
            break
        except Exception as e:
            print(f"[오류] {e}")
            break
    return False

def classify_query(index_text: Tuple[int, str], prompt: str) -> Tuple[int, bool]:
    index, text = index_text
    result = gpt_classify(text, prompt)
    return index, result

def process_directory(base_path: str, dir_name: str, prompt: str, output_dir: str):
    query_path = os.path.join(base_path, dir_name, "queries.jsonl")
    output_path = os.path.join(output_dir, f"{dir_name}.json")
    queries = read_jsonl(query_path)
    print(f"{dir_name}: {len(queries)}")

    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = (executor.submit(classify_query, q, prompt) for q in queries)
        for future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                print(f"[{dir_name}] 오류 발생: {e}")

    results.sort()
    with open(output_path, "w") as f:
        json.dump(results, f)

    total = sum([res for _, res in results])
    print(f"{dir_name}: {total} / {len(results)}")

if __name__ == "__main__":
    CONFIG = configparser.ConfigParser()
    CONFIG.read('config.ini')

    BASE = "/mnt/hlilabshare/jjunshim/data/beir"
    OUTPUT = "beir"
    os.makedirs(OUTPUT, exist_ok=True)

    PROMPT = """You are a binary classifier that detects whether a user’s question contains temporal references which may lead to different answers due to a knowledge cutoff.
    Check for words or phrases like “now,” “current,” “today,” “yesterday,” “tomorrow,” or explicit years (e.g., “in 2023”).

    - If the question contains such temporal references: output `True`.
    - Otherwise: output `False`.

    Examples:
    Input: "Who is the current president of South Korea?"
    Output: True

    Input: "What is the value of pi?"
    Output: False

    Input: "Who won the World Cup final yesterday?"
    Output: True

    Input: "Which planet is the largest in the Solar System?"
    Output: False
    """

    dirs = [d for d in os.listdir(BASE) if os.path.isdir(os.path.join(BASE, d))]

    with ProcessPoolExecutor(max_workers=4) as executor:
        for dir_name in dirs:
            executor.submit(process_directory, BASE, dir_name, PROMPT, OUTPUT)
