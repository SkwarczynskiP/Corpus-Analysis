import collections
import math
import re
import gensim.models
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from datasets import load_dataset
from gensim import corpora

dataset = load_dataset("yelp_review_full")


def normalize(text):
    # Lowercases each token
    text = text.lower()

    # Additional normalization due to the formatting of the reviews in the dataset
    # Gets rid of the "\n" characters in the dataset
    text = text.replace("\\n", "")

    # Removes all punctuation (referenced https://www.geeksforgeeks.org/python-remove-punctuation-from-string/)
    text = re.sub(r'[^\w\s]', '', text)

    # Experimentation normalization - one additional variation
    text = experimentation(text)

    # Lemmatization of each token (referenced
    # https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/ for different examples)
    lemmatizer = WordNetLemmatizer()
    tempList = text.split()
    lemma = [lemmatizer.lemmatize(tempList) for tempList in tempList]
    finalText = (' '.join(lemma))

    return finalText


def experimentation(text):
    stopWords = set(stopwords.words('english'))

    # Removes all stopwords
    tempList = text.split()
    filteredWords = [tempList for tempList in tempList if tempList.lower() not in stopWords]
    text = (' '.join(filteredWords))

    return text


def loglikelihoodCalculation(labelCount, vocab):
    loglikelihood = {}
    for label in labelCount.keys():
        loglikelihood[label] = {}
        denominator = sum(labelCount[label].values()) + len(vocab)

        # Plus one smoothing
        for word in vocab:
            numerator = labelCount[label][word] + 1
            probability = numerator / denominator
            loglikelihood[label][word] = math.log(probability)

    return loglikelihood


def main():
    # Dataset taken from: https://huggingface.co/datasets/yelp_review_full/viewer/yelp_review_full/train
    texts = dataset['train']['text'][0:1000]
    labels = dataset['train']['label'][0:1000]
    normalizedText = []

    actualLabels = ["1 Star Reviews", "2 Star Reviews", "3 Star Reviews", "4 Star Reviews", "5 Star Reviews"]

    # Normalizes the text from the dataset
    for text in texts:
        processed_text = normalize(text)
        normalizedText.append(processed_text)

    # Bag of words creation
    vocab = set()
    bow = []
    for text in normalizedText:
        wordCounts = {}
        tokens = nltk.word_tokenize(text)
        vocab.update(tokens)

        for word in tokens:
            if word in wordCounts:
                wordCounts[word] += 1
            else:
                wordCounts[word] = 1

        bow.append(wordCounts)

    # Initialization for gathering the word counts
    labelCount = {0: collections.defaultdict(int), 1: collections.defaultdict(int), 2: collections.defaultdict(int),
                  3: collections.defaultdict(int), 4: collections.defaultdict(int)}
    otherLabelCount = {0: collections.defaultdict(int), 1: collections.defaultdict(int),
                       2: collections.defaultdict(int), 3: collections.defaultdict(int),
                       4: collections.defaultdict(int)}

    # Word count for each label
    for x in range(len(bow)):
        wordCounts = bow[x]
        label = labels[x]

        for word, count in wordCounts.items():
            labelCount[label][word] += count

    # Word count for each of the other labels
    for x in range(len(bow)):
        wordCounts = bow[x]
        label = labels[x]

        # Label 0 = 1 star rating, index 1 = 2 star, index 2 = 3 star, index 3 = 4 star, index 4 = 5 star
        for word, count in wordCounts.items():
            if label == 0:
                count = labelCount[1][word] + labelCount[2][word] + labelCount[3][word] + labelCount[4][word]
                otherLabelCount[label][word] += count
            elif label == 1:
                count = labelCount[0][word] + labelCount[2][word] + labelCount[3][word] + labelCount[4][word]
                otherLabelCount[label][word] += count
            elif label == 2:
                count = labelCount[0][word] + labelCount[1][word] + labelCount[3][word] + labelCount[4][word]
                otherLabelCount[label][word] += count
            elif label == 3:
                count = labelCount[0][word] + labelCount[1][word] + labelCount[2][word] + labelCount[4][word]
                otherLabelCount[label][word] += count
            else:  # label == 4
                count = labelCount[0][word] + labelCount[1][word] + labelCount[2][word] + labelCount[3][word]
                otherLabelCount[label][word] += count

    # Printing out the average number of tokens per document per category
    tokensPerDocument = [0, 0, 0, 0, 0]
    documentCount = [0, 0, 0, 0, 0]
    print("\nAverage number of tokens per document per category:")
    for label, wordCounts in labelCount.items():
        if label == 0:
            tokensPerDocument[0] = sum(wordCounts.values())
        elif label == 1:
            tokensPerDocument[1] = sum(wordCounts.values())
        elif label == 2:
            tokensPerDocument[2] = sum(wordCounts.values())
        elif label == 3:
            tokensPerDocument[3] = sum(wordCounts.values())
        else:  # label == 4:
            tokensPerDocument[4] = sum(wordCounts.values())

    for x in range(len(labels)):
        if labels[x] == 0:
            documentCount[0] += 1
        elif labels[x] == 1:
            documentCount[1] += 1
        elif labels[x] == 2:
            documentCount[2] += 1
        elif labels[x] == 3:
            documentCount[3] += 1
        else:  # labels[x] == 4:
            documentCount[4] += 1

    for label in labelCount.keys():
        average = (tokensPerDocument[label] / documentCount[label])
        print(f"Average Tokens for {actualLabels[label]}: {average:.2f}")
    print("")


    # Begin Loglikelihood calculation

    # Calculation for the first term in the log likelihood equation: log(P(w|c))
    loglikelihoodFirstTerm = loglikelihoodCalculation(labelCount, vocab)

    # Calculation for the second term in the log likelihood equation: log(P(w|Co))
    loglikelihoodSecondTerm = loglikelihoodCalculation(otherLabelCount, vocab)

    # Calculation for the log likelihood ratios: log(P(w|c)) - log(P(w|Co))
    loglikelihood = {}
    for label in labelCount.keys():
        loglikelihood[label] = {}
        for token in vocab:
            loglikelihood[label][token] = loglikelihoodFirstTerm[label][token] - loglikelihoodSecondTerm[label][token]

    # Sort the loglikelihood ratios
    sortedLoglikelihoods = {}
    for label in labelCount.keys():
        sortedLoglikelihoods[label] = sorted(loglikelihood[label].items(), key=lambda item: item[1], reverse=True)

    # Print the top 10 words occurring in each label
    for label in labelCount.keys():
        print("Top 10 words in the label: " + actualLabels[label])
        for x in range(10):
            print(sortedLoglikelihoods[label][x])
        print("")
    # End of Loglikelihood calculation

    # Beginning of the LDA classification
    posts = [x.split(' ') for x in normalizedText]
    id2word = corpora.Dictionary(posts)
    corpus = [id2word.doc2bow(text) for text in posts]

    Lda = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=10, passes=2, workers=2)
    topics = []

    for idx, topic in Lda.print_topics(-1):
        words = topic.split("+")
        words = [word.strip().split("*")[1] for word in words]
        topics.append(words)
        print("Topic: {} \nWords: {}\n".format(idx, topic))

    # Finding the most dominant topics for each category (1 star, 2 star, 3 star, 4 star, 5 star reviews)
    mostDominantTopics = []
    for bow in corpus:
        topics = Lda.get_document_topics(bow)
        dominantTopic = max(topics, key=lambda i: i[1])[0]
        mostDominantTopics.append(dominantTopic)

    # Storing the topics by each individual category
    topics = {0: [], 1: [], 2: [], 3: [], 4: []}
    for x, topic in enumerate(mostDominantTopics):
        label = labels[x]
        topics[label].append(topic)

    # Gathering the top 3 topics for each category
    topicCounts = {label: collections.Counter(topics) for label, topics in topics.items()}
    topTopics = {label: count.most_common(3) for label, count in topicCounts.items()}

    # Printing the top 3 topics for each category
    print("The top three topics for each category:")
    for label, topics in topTopics.items():
        labelName = actualLabels[label]
        topics = [topic[0] for topic in topics]
        print(f"Label {label} ({labelName}): {', '.join(map(str, topics))}")
    # End of LDA classification


if __name__ == "__main__":
    main()

