from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
from tqdm import tqdm
import json
import os



def load_gsm8k(file_path:str):
    data=[]
    with open(file_path,"r",encoding="utf-8") as f:
        for line in f:
            item=json.loads(line)
            question = item['question']
            ground_truth=item['answer'].split("####")[1].strip()
            data.append({"question":question,"answer":ground_truth})
    return data
        
def build_prompt(question:str,strategy:str):
    final_prompt=""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if strategy =="question_only":
        prompt_path = os.path.join(current_dir, "prompts", "question_only.prompt")
        with open(prompt_path,"r",encoding="utf-8") as f:
            template=f.read()

        final_prompt=template.replace("{question}", question)
    elif strategy=="r1_zero":
        prompt_path = os.path.join(current_dir, "prompts", "r1_zero.prompt")
        with open(prompt_path,"r",encoding="utf-8") as f:
            template=f.read()

        final_prompt=template.replace("{question}", question)
    elif strategy=="r1_zero_three_shot":
        prompt_path = os.path.join(current_dir, "prompts", "r1_zero_three_shot_gsm8k.prompt")
        with open(prompt_path,"r",encoding="utf-8") as f:
            template=f.read()

        final_prompt=template.replace("{question}", question)
    else: 
        raise ValueError(f"Unknown strategy: {strategy}")

    return final_prompt

# 3. 针对不同策略，动态修改 sampling_params（关键！）
def get_sampling_params_for_strategy(strategy,sampling_params):
    params = dict(sampling_params)
    if strategy in ["r1_zero", "r1_zero_three_shot"]:
        params['stop'] = ["</answer>"]
        params['include_stop_str_in_output'] = True
    return params

# 4. 遍历数据集跑推理
def run_evaluation(data, strategy, sampling_params,batch_size=16):
    results = []

    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i+batch_size]
        
        # 拼接 Prompt
        prompts = [build_prompt(item['question'], strategy) for item in batch]
        
        # 获取对应的采样参数
        current_params = get_sampling_params_for_strategy(strategy,sampling_params)
        
        # 调用 vLLM
        completions = server.generate_completions(
            prompts=prompts, 
            sampling_params=current_params,
            batch_size=batch_size
        )
        
        # 将生成结果与 Ground Truth 对应保存
        for item, completion,prompt in zip(batch, completions,prompts):
            results.append({
                "prompt": prompt,
                "ground_truth": item['answer'],
                "model_output": completion.text # 或者 completion.token_ids
            })
    return results

def evaluate_and_count(result,strategy):
    category_1 = 0 # Format=1, Correct=1
    category_2 = 0 # Format=1, Correct=0 
    category_3 = 0 # Format=0, Correct=0 
    cat2_samples = []
    cat3_samples = []
  
    for res in result:
        if strategy =="question_only":
            fmt_reward, ans_reward, _ = question_only_reward_fn(res['model_output'], res['ground_truth'])
        else:
            fmt_reward, ans_reward, _ = r1_zero_reward_fn(res['model_output'], res['ground_truth'])

        if fmt_reward==1 and ans_reward==1:
            category_1+=1
    
        elif fmt_reward == 1 and ans_reward == 0:
            category_2 += 1
            if len(cat2_samples) < 10: 
                cat2_samples.append(res)
        elif fmt_reward == 0 and ans_reward == 0:
            category_3 += 1
            if len(cat3_samples) < 10:
                cat3_samples.append(res)


    print(f"类别1 (格式对+答案对): {category_1}")
    print(f"类别2 (格式对+答案错): {category_2}")
    print(f"类别3 (格式错+答案错): {category_3}")
    
    return cat2_samples, cat3_samples

if __name__=="__main__":

    data_path="/root/autodl-tmp/assignment5-alignment/data/gsm8k/test.jsonl"
    data=load_gsm8k(data_path)
    # 1. 启动 vLLM 服务
    local_path = "/root/autodl-tmp/assignment5-alignment/models/OLMo-2-0425-1B"
    server = VLLMServer(model_id=local_path, gpu=0)
    server.start()
    
    # 2. 准备采样参数
    sampling_params = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 512, 
        "n":1,
        "seed":42
    }

    # 3. 三种策略依次跑
    for strategy in ["question_only", "r1_zero", "r1_zero_three_shot"]:
        print(f"\n 开始评测: {strategy}")
        results = run_evaluation(data, strategy,sampling_params, batch_size=16)

        with open(f"results_{strategy}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        cat2, cat3 = evaluate_and_count(results, strategy)

        print("\n--- Category 2 样例 ---")
        for idx, sample in enumerate(cat2):
            print(f"样例 {idx+1}:")
            print(f"  Ground Truth: {sample['ground_truth']}")
            print(f"  Model Output: {sample['model_output']}...")

    server.stop()  # 跑完后关闭服务






