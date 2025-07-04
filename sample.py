import os
import json
import torch
from typing import List, Tuple
from tqdm import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


class TemporalClassifier:
    def __init__(self, model_id: str, device: str = "auto"):
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, device_map=device
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

    def classify(self, query: str, system_prompt: str) -> bool:
        """질문을 받아서 True/False로 판별"""
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": f"Now evaluate:\nInput: \"{query}\"\nOutput:"}]
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
            generated_tokens = outputs[0][input_len:]

        decoded = self.processor.decode(generated_tokens, skip_special_tokens=True).strip()

        # ✅ True/False 파싱
        normalized = decoded.lower()
        if "true" in normalized:
            return True
        elif "false" in normalized:
            return False
        else:
            print(f"[경고] 예상 못한 출력: {decoded}")
            return False  # 안전하게 기본 False로 처리

    @staticmethod
    def read_jsonl(filepath: str) -> List[Tuple[int, str]]:
        with open(filepath, 'r', encoding='utf-8') as f:
            return [(i, json.loads(line.strip())["text"]) for i, line in enumerate(f) if line.strip()]

    @staticmethod
    def save_to_jsonl(filepath: str, index: int, label: bool):
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": index, "label": label}, ensure_ascii=False) + "\n")


def main():
    BASE_DIR = "/mnt/hlilabshare/khyunjin1993/tpour/"
    OUTPUT_DIR = "data"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_id = "google/gemma-3-12b-it"
    classifier = TemporalClassifier(model_id=model_id, device="cuda:3")

    PROMPT = """You are a binary classifier that detects whether a sentence contains temporal characteristics such as changes over time, trends, specific time points, durations, or temporal sequences.

- If the sentence contains any temporal characteristics: output `True`.
- Otherwise: output `False`.

Examples:
Input: "The popularity of this topic has steadily grown."
Output: True

Input: "The algorithm sorts items alphabetically."
Output: False

Input: "User behavior changes significantly during holiday seasons."
Output: True

Input: "The model uses a standard transformer encoder."
Output: False
"""

    dirs = [d for d in os.listdir(BASE_DIR)]

    for dir_name in dirs:
        query_path = os.path.join(BASE_DIR, dir_name)
        output_path = os.path.join(OUTPUT_DIR, f"{dir_name}.jsonl")

        queries = classifier.read_jsonl(query_path)
        print(f"Processing {dir_name}... ({len(queries)} queries)")

        for idx, query in tqdm(queries, desc=f"[{dir_name}] 진행중", unit="건"):
            label = classifier.classify(query, PROMPT)
            classifier.save_to_jsonl(output_path, idx, label)

        print(f"Finished {dir_name}\n")


if __name__ == "__main__":
    main()
