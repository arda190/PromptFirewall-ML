from gensim.models import Word2Vec

sentences = [
    # Original Data
    ["i", "love", "machine", "learning"],
    ["machine", "learning", "is", "awesome"],
    ["i", "love", "python"],
    ["python", "is", "powerful"],

    # AI & Deep Learning
    ["deep", "learning", "uses", "neural", "networks"],
    ["neural", "networks", "learn", "from", "data"],
    ["artificial", "intelligence", "is", "transforming", "technology"],
    ["transformers", "are", "popular", "in", "nlp"],
    ["large", "language", "models", "generate", "text"],

    # Python & Programming
    ["python", "has", "simple", "syntax"],
    ["developers", "write", "clean", "code"],
    ["functions", "return", "values", "in", "python"],
    ["object", "oriented", "programming", "is", "useful"],
    ["debugging", "code", "takes", "time"],

    # Data Science & Engineering
    ["data", "scientists", "analyze", "large", "datasets"],
    ["pandas", "is", "great", "for", "data", "manipulation"],
    ["clean", "data", "improves", "model", "performance"],
    ["visualization", "helps", "understand", "data", "patterns"],
    ["feature", "engineering", "boosts", "accuracy"]
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    workers=4,
)


print(model.wv.most_similar("machine"))
