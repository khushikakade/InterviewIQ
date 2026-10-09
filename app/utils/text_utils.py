import re
import math
from collections import Counter

FILLER_WORDS = {
    'um', 'uh', 'like', 'actually', 'basically', 'you know', 'so',
    'literally', 'honestly', 'right', 'mean', 'sort of', 'kind of'
}

TECHNICAL_KEYWORDS_BANK = {
    'python', 'java', 'c++', 'sql', 'javascript', 'html', 'css', 'react', 'flask',
    'django', 'node', 'express', 'postgresql', 'mysql', 'sqlite', 'mongodb',
    'aws', 'azure', 'docker', 'kubernetes', 'git', 'github', 'ci/cd',
    'machine learning', 'deep learning', 'nlp', 'opencv', 'mediapipe', 'pytorch',
    'tensorflow', 'scikit-learn', 'pandas', 'numpy', 'data structures', 'algorithms',
    'object-oriented', 'system design', 'rest api', 'microservices', 'agile'
}

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize(text):
    clean = clean_text(text).lower()
    return re.findall(r'\b[a-z0-9+#.]+\b', clean)

def count_words(text):
    tokens = tokenize(text)
    return len(tokens)

def detect_filler_words(text):
    clean_lower = text.lower()
    found_fillers = {}
    total_count = 0
    
    for filler in FILLER_WORDS:
        # Match whole phrase or word
        pattern = r'\b' + re.escape(filler) + r'\b'
        matches = re.findall(pattern, clean_lower)
        if matches:
            found_fillers[filler] = len(matches)
            total_count += len(matches)
            
    return total_count, found_fillers

def calculate_vocabulary_richness(text):
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    unique_tokens = set(tokens)
    # Type-Token Ratio (TTR) normalized
    return round((len(unique_tokens) / len(tokens)) * 100, 1)

def compute_cosine_similarity(text1, text2):
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)
    if not tokens1 or not tokens2:
        return 0.0
    
    vec1 = Counter(tokens1)
    vec2 = Counter(tokens2)
    
    intersection = set(vec1.keys()) & set(vec2.keys())
    dot_product = sum([vec1[x] * vec2[x] for x in intersection])
    
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    if not denominator:
        return 0.0
    
    return round((dot_product / denominator) * 100, 1)

def extract_keywords(text):
    tokens = tokenize(text)
    found = [t for t in set(tokens) if t in TECHNICAL_KEYWORDS_BANK or len(t) > 4]
    return list(set(found))
