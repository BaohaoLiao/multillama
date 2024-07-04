from tqdm import tqdm
import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


LANG_TABLE = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "cs": "Czech",
    "is": "Icelandic",
    "zh": "Chinese",
    "ja": "Japanese",
    "ru": "Russian",
    "uk": "Ukrainian",
    "ha": "Hausa",
    "ro": "Romanian",
    "gu": "Gujarati",
    "hi": "Hindi",
    "es": "Spanish"
}

def get_key_suffix(tgt_lang):
    return f"\n{LANG_TABLE[tgt_lang]}:"
    
def clean_outputstring(output, key_word, split_idx):
    try:
        out = output.split(key_word)[split_idx].split("\n")
        if out[0].strip() != "":
            return out[0].strip()
        elif out[1].strip() != "":
            ## If there is an EOL directly after the suffix, ignore it
            print(f"Detect empty output, we ignore it and move to next EOL: {out[1].strip()}")
            return out[1].strip()
        else:
            print(f"Detect empty output AGAIN, we ignore it and move to next EOL: {out[2].strip()}")
            return out[2].strip()
    except:
        print(f"Can not recover the translation by moving to the next EOL.. Trying move to the next suffix")
        
    try:
        return output.split(key_word)[2].split("\n")[0].strip()
    except:
        print(f"Can not solve the edge case, recover the translation to empty string! The output is {output}")
        return ""
    
def get_prompt(src_lang, tgt_lang, src_sent):
    prefix = f"Translate this from {LANG_TABLE[src_lang]} to {LANG_TABLE[tgt_lang]}:\n"
    src_sent = f"{LANG_TABLE[src_lang]}: {src_sent}"
    suffix = f"\n{LANG_TABLE[tgt_lang]}:"
    return prefix, src_sent, suffix

def load_model(base_model, peft_model, max_source_length, max_new_tokens):
    config_kwargs = {
        "trust_remote_code": True,
        "max_length":max_source_length + max_new_tokens,
    }
    config = AutoConfig.from_pretrained(base_model, **config_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map="cuda",
    )
    model.generation_config.max_length = max_source_length + max_new_tokens
    model.generation_config.use_cache = True
    model = PeftModel.from_pretrained(model, peft_model)

    model = model.merge_and_unload()
    model.eval()    
    return model


def load_tokenizer(base_model):
    tokenizer_kwargs = {
        "padding_side": 'left',
        "add_eos_token": False,
    }
    tokenizer = AutoTokenizer.from_pretrained(base_model,**tokenizer_kwargs,)
 
    if "Llama-3" in base_model:
        tokenizer.add_special_tokens(dict(pad_token="<|padding|>"))
    elif "Mistral" in base_model:
        tokenizer.pad_token_id = 0
    return tokenizer


def main(args):
    src_lang, tgt_lang = args.lang_pair.split("-")[0], args.lang_pair.split("-")[1] 
    file_path = f"{args.input_dir}/wmttest2024.txt.{args.lang_pair}.{src_lang}"
    save_path = f"{args.output_dir}/test-{args.lang_pair}"
    suffix = get_key_suffix(tgt_lang)
    split_idx = 1

    with open(file_path, 'r', encoding="utf-8") as file:
        src_sents = [line.strip() for line in file]

    model = load_model(args.base_model, args.peft_model, args.max_source_length, max_new_tokens)
    tokenizer = load_tokenizer(args.base_model)

    count = 0
    tgt_sents = []
    for src_sent in tqdm(src_sents):
        tag_prefix, tag_src_sent, tag_suffix = get_prompt(src_lang, tgt_lang, src_sent)
        prompt = tag_prefix + tag_src_sent + tag_suffix
        input_ids = tokenizer(prompt, return_tensors="pt", padding=True, max_length=args.max_source_length, truncation=True).input_ids
        if input_ids.shape[1] <= args.max_source_length:
            tag_src_input_ids = tokenizer(tag_src_sent, return_tensors="pt").input_ids[0]
            num_deletion = input_ids.shape[1] - args.max_source_length
            tag_src_sent = tokenizer.decode(tag_src_input_ids[:-(num_deletion+1)], skip_special_tokens=True)
            prompt = tag_prefix + tag_src_sent + tag_suffix
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        
        max_new_tokens = min(args.max_new_tokens, int(input_ids.shape[1] * args.length_ratio))
        #with torch.no_grad():
        with torch.cuda.amp.autocast():
            generated_ids = model.generate(
                input_ids=input_ids.to(args.device), 
                num_beams=args.num_beams, 
                max_new_tokens=max_new_tokens, 
                do_sample=True, 
                temperature=0.6, 
                top_p=0.9
            )
            if max_new_tokens + input_ids.shape[1] == generated_ids.shape[1]:
                generated_ids = model.generate(
                    input_ids=input_ids.to(args.device), 
                    num_beams=1, 
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=1.5,
                    do_sample=True, 
                    temperature=0.6, 
                    top_p=0.9
                )

        decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        pred = clean_outputstring(decoded_preds, suffix, split_idx)
        print("------------------")
        print(max_new_tokens, generated_ids.shape[1] - input_ids.shape[1], input_ids.shape[1])
        print(decoded_preds)
        print(pred)

        #if (lang_pair == "en-zh") or (lang_pair == "ja-zh"):
        #    tgt_sents.append(finalize_chinese_text(pred))
        #else:
        tgt_sents.append(pred)


        count += 1
        if count > 10:
            break

    """
    assert len(src_sents) == len(tgt_sents)
    for i in range(len(src_sents)):
        if tgt_sents[i] == "":
            tgt_sents[i] == src_sents[i]
    """

    with open(save_path, 'w', encoding='utf-8') as file:
        for tgt_sent in tgt_sents:
            file.write(tgt_sent.strip() + '\n')


def arg_parse():
    parser = argparse.ArgumentParser(description="Generation")
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--peft_model", type=str)
    parser.add_argument("--lang_pair", type=str)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--length_ratio", type=float, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=5)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parse()
    print(args)
    main(args)
    