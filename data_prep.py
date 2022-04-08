import re
from transformers import BertTokenizer


def text_cleaner(text):
    newString = text.lower()
    newString = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', newString) 
    newString = re.sub('[^a-zA-Z#@]', ' ', newString)
    return newString


def tokenize_tweet(tweet, bert_variant="bert-base-uncased", seq_len=128):
    tweet = text_cleaner(tweet)
    tokenizer = BertTokenizer.from_pretrained(bert_variant)
    tokenizer_output = tokenizer(tweet, max_length = seq_len, padding = "max_length", return_tensors="pt")
    input_ids, attn_mask = tokenizer_output["input_ids"], tokenizer_output["attention_mask"]
    return input_ids, attn_mask
