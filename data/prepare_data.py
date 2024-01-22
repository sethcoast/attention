
def write_data(file_path, my_list):
    # write en sentences
    with open(file_path, "w", encoding="utf-8") as file:
        for item in my_list:
            file.write(item + "\n")

def load_data(file_path, en_save_path, ja_save_path):
    # split english and japanese sentences into separate lists
    english_sentences, japenese_sentences = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            en, ja = line.strip().split('\t')
            english_sentences.append(en)
            japenese_sentences.append(ja)
    
    # write english and japanese sentences to files
    write_data(en_save_path, english_sentences)
    write_data(ja_save_path, japenese_sentences)

load_data("raw/raw", "stage/english", "stage/japanese")
    
    