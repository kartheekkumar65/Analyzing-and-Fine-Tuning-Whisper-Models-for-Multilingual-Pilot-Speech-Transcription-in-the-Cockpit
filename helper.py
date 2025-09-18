import spacy
import re
from word2number import w2n
# import translators as ts
# from deep_translator import GoogleTranslator
from whisper_normalizer.english import EnglishTextNormalizer, EnglishNumberNormalizer, EnglishSpellingNormalizer
# python -m spacy download en_core_web_sm

"""
Normalizes Text
"""
# Load the English language model
nlp = spacy.load("en_core_web_sm")

# NATO Alphabet dictionary for letter conversion
nato_alphabet = {
    'ALPHA': 'A', 'BRAVO': 'B', 'CHARLIE': 'C', 'DELTA': 'D', 'ECHO': 'E', 'FOXTROT': 'F', 'FOX': 'F','GOLF': 'G',
    'HOTEL': 'H', 'INDIA': 'I', 'JULIETT': 'J', 'KILO': 'K', 'LIMA': 'L', 'MIKE': 'M', 'NOVEMBER': 'N',
    'OSCAR': 'O', 'PAPA': 'P', 'QUEBEC': 'Q', 'ROMEO': 'R', 'SIERRA': 'S', 'TANGO': 'T', 'UNIFORM': 'U',
    'VICTOR': 'V', 'WHISKEY': 'W', 'XRAY': 'X', 'YANKEE': 'Y', 'ZULU': 'Z'
}

# List of filler words
fill_words = ["um", "uh", "like", "you know", "actually", "basically", "seriously", "ähm", "ach", "mh", "mhm"]

def convert_number_words(text):
    # This function converts number words to numeric values (e.g. "seven thousand" -> "7000"
    split_text = text.split()
    words = []
    for word in split_text:
        try:
            words += [str(w2n.word_to_num(word))]
        except ValueError:
            words += [word] 
    text=' '.join(words)
    return text

def normalize_combined_words(text):
    # 1. NATO-Alphabet Wörter wie "DELTA", "LIMA", "ROMEO" zu "DLR" zusammenführen
    for word, letter in nato_alphabet.items():
        text = re.sub(rf'\b{word}\b', letter, text, flags=re.IGNORECASE)

    # 2. Erkennung von getrennten Buchstaben, die ein Akronym sein könnten (z.B. "D L R" -> "DLR")
    text = re.sub(r'\b([A-Z]) [ -]?([A-Z]) [ -]?([A-Z])\b', r'\1\2\3', text)

    # 3. Erkennung von getrennten Wörtern, die zusammengeschrieben sein könnten (z.B. "take off", "take-off")
    muster = re.findall(r'\b(\w+)[ -]?(\w+)\b', text)

    for wort1, wort2 in muster:
        zusammen = wort1 + wort2  # Kombinierte Version
        getrennt = wort1 + " " + wort2  # Trennbare Version
        mit_bindestrich = wort1 + "-" + wort2  # Mit Bindestrich

        # Überprüfen, ob eine Variante im Text existiert und ersetze mit der zusammengefügten Version
        if zusammen in text:
            ersatz = zusammen
        elif mit_bindestrich in text:
            ersatz = zusammen  # Wenn mit Bindestrich, dann ohne Bindestrich zusammenfügen
        else:
            ersatz = getrennt  # Standard: Lassen Sie das Wort mit Leerzeichen zusammen

        # Ersetze das gefundene Muster durch die sinnvollste Variante
        text = re.sub(rf"\b{wort1}[ -]?{wort2}\b", ersatz, text, flags=re.IGNORECASE)

    return text

# Function to filter out fill words from a text
def filter_fill_words(text, fill_words):
    # Convert the text to lowercase and split into words
    words = text.lower().split()
    
    # Filter out fill words
    filtered_words = [word for word in words if word not in fill_words]
    
    # Rejoin the words into a cleaned-up text
    return ' '.join(filtered_words)

def transform_nato_alphabet(text):
    words = text.split()
    normalized_words = []
    
    for word in words:
        word_stripped = re.sub(r'[^\w\s]', '', word)  # Remove punctuation
        
        # Check if the word is in the NATO alphabet
        if word_stripped.upper() in nato_alphabet:
            normalized_words.append(nato_alphabet[word_stripped.upper()] + word[len(word_stripped):])
        else:
            normalized_words.append(word)
    
    # Join the words back together
    return ' '.join(normalized_words)

def proposed_1(text):
    # Convert to lowercase
    text = text.lower()

    # Translate text (if needed)
    #translated_text = GoogleTranslator(source='de', target='en').translate(text)
    #print(translated_text)

    # Normalize NATO alphabet
    nato_text = transform_nato_alphabet(text)
    #print(nato_text)

    combined_words_text = normalize_combined_words(nato_text)

    # Normalize numbers (for words like "seven thousand" to "7000")
    # number_norm_text = convert_number_words(combined_words_text)
    # print(number_norm_text)

    # Lemmatization with spaCy
    doc = nlp(combined_words_text)
    lemmatized_text = " ".join([token.lemma_ for token in doc if not token.is_punct])
    #print(lemmatized_text)
    
    # Filter out fill words
    cleaned_text = filter_fill_words(lemmatized_text, fill_words)
    #print(cleaned_text)

    return cleaned_text

def proposed_2(text):
    whisper_normalized_text = whisper_normalize(text)
    final_text = proposed_1(whisper_normalized_text)
    
    return final_text
    
def proposed_3(text):
    
    proposed_1_text = proposed_1(text)
    final_text = whisper_normalize(proposed_1_text)
    
    return final_text


def number_normalize(text):
    numberNormalizer = EnglishNumberNormalizer()
    normalized_text = numberNormalizer(text)

    return normalized_text
    
def whisper_normalize(text):
    textNormalizer = EnglishTextNormalizer()
    numberNormalizer = EnglishNumberNormalizer()
    spellingNormalizer = EnglishSpellingNormalizer()
    textNorm = textNormalizer(text)
    numberNorm = numberNormalizer(textNorm)
    normalized_text = spellingNormalizer(numberNorm)

    #print(normalized_verified_text)
    #print(normalized_whisper_text)

    return normalized_text
