import openai
import pandas as pd
from pathlib import Path
import jsonlines
import sys
import os
from prompt import prompt_template, task_description, example_list

def extract_answer(x):
    out = x.split("the answer is:")
    if len(out) == 1:
        return None
    answer = out[1].strip(" ").split(" ")[0].strip(".").replace(",", "")
    if answer.endswith("%"):
        return float(answer.strip("%")) / 100.0
    else:
        try:
            return float(answer)
        except:
            return None

output_file = sys.argv[1]
openai.api_key = os.getenv("OPENAI_API_KEY")
test_file = "./grade-school-math/grade_school_math/data/test.jsonl"
if not Path(test_file).exists():
    os.system("git clone https://github.com/openai/grade-school-math.git")

test_qa_list = []
with jsonlines.open(test_file) as f:
    for line in f.iter():
        test_qa_list.append(line)
gsm8k_prompt = prompt_template.format(task=task_description, examples="\n".join(example_list))
records = []
bs = 20
qa_list = test_qa_list
n = (len(qa_list) // bs) + 1
for idx in range(n):
    print(idx)
    sub_list = qa_list[(idx*bs):((idx+1)*bs)]
    prompts = [gsm8k_prompt + "Question: " + qa["question"] + "\nAnswer:" for qa in sub_list]
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompts,
        max_tokens=450,
        temperature=0
    )
    for i in range(len(sub_list)):
        records.append([sub_list[i]["question"], sub_list[i]["answer"], response["choices"][i]["text"]])

df = pd.DataFrame.from_records(records, columns=["question", "ground_truth", "model_output"])
df["ground_truth_answer"] = df["ground_truth"].apply(lambda x: float(x.split("### ")[1].replace(",", "")))
df["model_answer"] = df["model_output"].apply(extract_answer)
correct_df = df[(df["ground_truth_answer"] - df["model_answer"]).abs() < 1e-5]
print("correct percentage: {0:0.1f}".format(100*correct_df.shape[0]/df.shape[0]))
df.to_csv(output_file, index=False)
