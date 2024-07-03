import fire
import regex
from tqdm import tqdm

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
    return f"Translate this from {LANG_TABLE[src_lang]} to {LANG_TABLE[tgt_lang]}:\n{LANG_TABLE[src_lang]}: {src_sent}\n{LANG_TABLE[tgt_lang]}:"

def load_model(base_model, peft_model, max_source_length, max_new_tokens):
    config_kwargs = {
        "trust_remote_code": True,
        "max_length":max_source_length + max_new_tokens,
    }
    config = AutoConfig.from_pretrained(base_model, **config_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        config=config,
        torch_dtype="auto",
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

def finalize_chinese_text(text):
    # This regex modifies the text by:
    # 1. Removing spaces between Han characters and digits, digits and Han characters, Han characters and emojis,
    #    emojis and Han characters, before the first English word, and after the last English word surrounded by Han characters
    # 2. Removing spaces before Chinese punctuation
    # 3. Replacing English periods with Chinese periods within the context of Chinese text
    pattern = (
        r'(?<!@[\w\d]+)\s+(?=\p{IsHan}|\d)|'
        r'(?<=\p{IsHan}|\d%)\s+(?!(?<=@\w+)\s)(?=\d|\p{IsHan})|'
        r'(?<=\p{IsHan})\s+(?=\p{Emoji})|'
        r'(?<=\p{Emoji})\s+(?=\p{IsHan})|'
        r'(?<=\p{IsHan})\s+(?=[A-Za-z])|'
        r'(?<=[A-Za-z])\s+(?=\p{IsHan})|'
        r'(?<=.)\s+(?=。|，|：|；|？|！|”|“|【|】|》|《)'
    )
    cleaned_text = regex.sub(pattern, '', text)
    # Replace English periods only if surrounded by Chinese characters or at the end of a sentence before Chinese punctuation
    cleaned_text = regex.sub(r'(?<=\p{IsHan})\.+(?=\p{IsHan}|[。，：；？！]?$)', '。', cleaned_text)
    return cleaned_text


def main(
        base_model: str, 
        peft_model: str, 
        lang_pair: str,
        input_dir: str,
        output_dir: str,
        length_ratio=2,
        max_source_length=512,
        max_new_tokens=512,
        device="cuda",
        num_beams=5,
    ):
    print("hello")
    src_lang, tgt_lang = lang_pair.split("-")[0], lang_pair.split("-")[1] 
    file_path = f"{input_dir}/wmttest2024.txt.{lang_pair}.{src_lang}"
    save_path = f"{output_dir}/test-{lang_pair}"
    suffix = get_key_suffix(tgt_lang)
    split_idx = 1

    with open(file_path, 'r', encoding="utf-8") as file:
        src_sents = [line.strip() for line in file]

    model = load_model(base_model, peft_model, max_source_length, max_new_tokens)
    tokenizer = load_tokenizer(base_model)

    tgt_sents = []
    for src_sent in tqdm(src_sents):
        prompt = get_prompt(src_lang, tgt_lang, src_sent)
        input_ids = tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            max_length=max_source_length,
            truncation=True).input_ids.to(device)

        max_new_tokens = min(max_new_tokens, int(input_ids.shape[1] * length_ratio))
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids, 
                num_beams=num_beams, 
                max_new_tokens=max_new_tokens, 
                #do_sample=True, 
                #temperature=0.6, 
                #top_p=0.9
            )
            if max_new_tokens + input_ids.shape[1] == generated_ids.shape[1]:
                generated_ids = model.generate(
                    input_ids=input_ids, 
                    num_beams=1, 
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=1.5
                )

        decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        pred = clean_outputstring(decoded_preds, suffix, split_idx)

        if (lang_pair == "en-zh") or (lang_pair == "ja-zh"):
            tgt_sents.append(finalize_chinese_text(pred))
        else:
            tgt_sents.append(pred)

    assert len(src_sents) == len(tgt_sents)
    for i in range(len(src_sents)):
        if tgt_sents[i] == "":
            tgt_sents[i] == src_sents[i]

    with open(save_path, 'w', encoding='utf-8') as file:
        for tgt_sent in tgt_sents:
            file.write(tgt_sent.strip() + '\n')


if __name__ == "__name__":
    fire.Fire(main)