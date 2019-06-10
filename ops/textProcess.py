import nltk

def a():
    document = 'the petals of the flower are pink in color and have a yellow center.'
    sentences = nltk.sent_tokenize(document)
    for sent in sentences:
        print(nltk.pos_tag(nltk.word_tokenize(sent)))
    
