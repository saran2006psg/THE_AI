from collections import Counter

def get_pairs(words):
    pairs = Counter()

    for word in words:
        for i in range(len(word)-1):
            pairs[(word[i], word[i+1])] += 1

    return pairs


def merge_pair(words, pair):
    merged_words = []

    for word in words:
        new_word = []
        i = 0

        while i < len(word):
            if (
                i < len(word)-1 and
                word[i] == pair[0] and
                word[i+1] == pair[1]
            ):
                new_word.append(pair[0] + pair[1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        merged_words.append(new_word)

    return merged_words