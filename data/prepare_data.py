import MeCab
import nltk
nltk.download('punkt')
import math
import sentencepiece as spm

from nltk.tokenize import word_tokenize


def write_data(file_path, my_list):
    # write en sentences
    with open(file_path, "w", encoding="utf-8") as file:
        for item in my_list:
            file.write(item + "\n")

def load_data(file_path, en_save_path=None, ja_save_path=None, num_lines=math.inf):
    # split english and japanese sentences into separate lists
    english_sentences, japenese_sentences = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            en, ja = line.strip().split('\t')
            english_sentences.append(en)
            japenese_sentences.append(ja)
            
            # optional number of lines to load to limit size of dataset
            if i == num_lines:
                break
    
    # write english and japanese sentences to files
    write_data(en_save_path, english_sentences) if en_save_path else None
    write_data(ja_save_path, japenese_sentences) if ja_save_path else None

    return english_sentences, japenese_sentences

# Function to tokenize Japanese text
def tokenize_japanese(text):
    # Initialize MeCab
    tagger = MeCab.Tagger()
    
    node = tagger.parseToNode(text)
    tokens = []
    while node:
        if node.surface != "":
            tokens.append(node.surface)
        node = node.next
    return tokens

def tokenize_english(text):
    return word_tokenize(text)

# what are the actual steps??? 
# What is the end goal desired product?
# I want 


if __name__ == "__main__":
    # load the data into separate lists
    en_path = "data/stage/english.txt"
    ja_path = "data/stage/japanese.txt"
    english_sentences, japenese_sentences = load_data("data/split/test", en_path, ja_path)

    ## train BPE on each corpora (vocabulary size of 30,000)
    spm.SentencePieceTrainer.train(input=en_path, model_prefix='en_bpe', vocab_size=2000)
    spm.SentencePieceTrainer.train(input=ja_path, model_prefix='jp_bpe', vocab_size=2000)

    ## apply BPE to each corpora
    # load the trained models (necessary? or do the above lines return the models?)
    sp_en = spm.SentencePieceProcessor(model_file='en_bpe.model')
    sp_jp = spm.SentencePieceProcessor(model_file='jp_bpe.model')

    # apply the models to the corpora (encode the sentences)
    encoded_en = [sp_en.encode(english_sentence, out_type=int) for english_sentence in english_sentences]
    encoded_jp = [sp_jp.encode(japanese_sentence, out_type=int) for japanese_sentence in japenese_sentences]
    