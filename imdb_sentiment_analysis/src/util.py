import os
import re

def get_path(in_root, in_folder):
    return os.path.join(in_root, in_folder)

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

def clean_text(text):
    text_re = re.compile('<.*?>')
    text = re.sub(text_re, '', text)
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = [word for word in text.split(" ") if not word in stop_words]
    text = " ".join(text)
    return text








