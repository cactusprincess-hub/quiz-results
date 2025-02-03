import torch
from transformers import GPT2Tokenizer, GPT2Model
from tqdm import tqdm

def calculate_embeddings(model, tokenizer, sequence, device):
    # 设置填充标记
    tokenizer.pad_token = tokenizer.eos_token  # 使用eos_token作为填充标记
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # 如果你想添加新的pad_token，可以取消此行注释

    # Tokenize输入序列并返回张量
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # 将数据移动到设备

    # 获取模型的嵌入输出
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 提取嵌入
    embeddings = outputs.last_hidden_state.mean(dim=1)  # 取每个token的平均作为序列的嵌入向量
    return embeddings

def main():
    model_name = "gpt2"  # 预训练的模型名称（可以根据需要更改）
    model = GPT2Model.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # 将模型移动到设备（如果有CUDA设备则使用GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 读取输入的FASTA文件
    fasta_file = "C:/Users/HP/Desktop/bit/plasmids.fasta"
    output_file = "C:/Users/HP/Desktop/new_embeddings_output.txt"

    with open(fasta_file, "r") as f:
        sequences = f.read().splitlines()

    # 处理每个序列并计算嵌入
    embeddings_list = []
    for sequence in tqdm(sequences, desc="Processing sequences"):
        embeddings = calculate_embeddings(model, tokenizer, sequence, device)
        embeddings_list.append(embeddings.cpu().numpy())

    # 将嵌入结果保存到文件
    with open(output_file, "w") as out_file:
        for embedding in embeddings_list:
            out_file.write(" ".join(map(str, embedding[0])) + "\n")

    print(f"Embeddings have been saved to {output_file}")

if __name__ == "__main__":
    main()




