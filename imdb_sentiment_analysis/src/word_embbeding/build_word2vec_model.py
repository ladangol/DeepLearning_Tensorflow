from src.util import get_path
import csv
import logging

def clean_text(text):
    stop_words = ["is", "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
                  "you", "your", "yours", "yourself", "yourselves", "he", "him", "his",
                  "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they",
                  "them", "their", "theirs", "themselves", "what", "which", "who", "whom",
                  "this", "that", "these", "those", "am", "is", "are", "was", "were", "be",
                  "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing",
                  "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
                  "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
                  "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out",
                  "on", "off", "over", "under", "again", "further", "then", "once", "here", "there",
                  "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
                  "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
                  "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
    import re
    text_re = re.compile('<.*?>')
    text = re.sub(text_re, '', text)
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = [word for word in text.split(" ") if not word in stop_words]
    text = " ".join(text)
    return text


def prepare_data(in_config):
    print("Start preparing data!")
    file_name = get_path(in_config.data_path_root, in_config.data_file)
    reader = csv.DictReader(open(file_name, encoding="utf-8"))
    list_reader = list(reader)

    data = [e['review'] for e in list_reader]
    print("data set size:", len(data))

    clean_data = list(map(clean_text, data))
    clean_data = [s.split() for s in clean_data]

    print("Data preparation completed!")
    return clean_data

def build_model(in_config):
    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', \
                        level=logging.INFO)

    clean_sentences = prepare_data(in_config)

    # Initialize and train the model (this will take some time)
    from gensim.models.word2vec import Word2Vec
    print("Training model...")
    model = Word2Vec(clean_sentences, workers=in_config.num_workers, \
                              size=in_config.num_features, min_count=in_config.min_word_count, \
                              window=in_config.context_window, sample=in_config.downsampling)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    print('saving model')
    model_name = get_path(in_config.data_path_root, in_config.word2vec_model)
    model.save(model_name)
    return model