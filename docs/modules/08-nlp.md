# Module 8: Natural Language Processing

## Introduction

Today we tackle natural language processing—teaching machines to understand and generate text.

Text is everywhere in business: customer reviews, support tickets, emails, social media, contracts, reports. Being able to automatically classify, extract information from, and generate text is incredibly valuable.

In Module 7, we saw how CNNs revolutionized image processing. Today, we'll see how transformers revolutionized NLP. The transformer architecture—introduced in 2017—is the foundation for BERT, GPT, and essentially every language model you've heard of.

By the end of this module, you'll understand how text becomes numbers, why transformers work so well, and how to leverage pre-trained models for your own applications.

**What is machine "understanding"?** Machines don't understand text like humans—they operate on statistical representations where similar meanings cluster together. What we call "understanding" is sophisticated pattern matching: a model that predicts masked words correctly has learned syntax, semantics, and world knowledge encoded as neural network weights. Whether this constitutes "understanding" or merely simulates it remains philosophically contested.

---

## Learning Objectives

By the end of this module, you should be able to:

1. **Explain** different text representation methods (BoW, TF-IDF, embeddings)
2. **Understand** why word order and context matter in NLP
3. **Describe** RNN architecture and the vanishing gradient problem
4. **Explain** the transformer architecture and self-attention mechanism
5. **Apply** pre-trained language models (BERT, GPT) for NLP tasks
6. **Identify** appropriate NLP approaches for business problems

---

## 8.1 Text Representation

### The Challenge of Text

Text is fundamentally different from tabular data:
- **Variable length**: Sentences can be 5 words or 500
- **Order matters**: "Dog bites man" ≠ "Man bites dog"
- **Same word, different meanings**: "bank" (river) vs "bank" (financial)
- **Vast vocabulary**: Hundreds of thousands of words

**Goal**: Convert text to numerical vectors that capture meaning.

### Bag of Words (BoW)

The simplest approach: count word occurrences.

| Document | "love" | "machine" | "learning" |
|----------|--------|-----------|------------|
| "I love machine learning" | 1 | 1 | 1 |
| "Machine learning is great" | 0 | 1 | 1 |

Each document becomes a vector of word counts.

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "I love machine learning",
    "Machine learning is great",
    "I love deep learning"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out())
# ['deep', 'great', 'is', 'learning', 'love', 'machine']
```

**Limitations:**
- Ignores word order: "dog bites man" = "man bites dog"
- Sparse and high-dimensional
- No semantic similarity: "good" and "great" are unrelated

**Think of it like a recipe vs. a shopping list**: BoW gives you the ingredients (flour, eggs, sugar, butter) but loses the recipe (the order and method matter!). "Cream butter and sugar, then add eggs" produces cake. "Add eggs, then cream butter and sugar" produces scrambled eggs with a butter problem. Same ingredients, completely different outcomes. BoW can't tell these apart.

> **Numerical Example: BoW Sparsity Problem**
>
> ```python
> # Compare BoW vs embedding representations
> vocab_size = 10000
> embedding_dim = 300
> sentence_words = 4  # "I love machine learning"
>
> # BoW: huge sparse vector
> bow_nonzero = sentence_words
> bow_sparsity = (vocab_size - bow_nonzero) / vocab_size * 100
> bow_memory = vocab_size * 4 / 1024  # KB (float32)
>
> # Embeddings: small dense vector
> emb_memory = embedding_dim * 4 / 1024  # KB
>
> print(f"BoW vector:       {vocab_size:,} dims, {bow_nonzero} non-zero, {bow_sparsity:.2f}% sparse")
> print(f"Embedding vector: {embedding_dim} dims, all non-zero, 0% sparse")
> print(f"Memory: BoW={bow_memory:.1f} KB vs Embedding={emb_memory:.1f} KB")
> print(f"Dimensionality reduction: {vocab_size/embedding_dim:.0f}x")
> ```
>
> **Output:**
> ```
> BoW vector:       10,000 dims, 4 non-zero, 99.96% sparse
> Embedding vector: 300 dims, all non-zero, 0% sparse
> Memory: BoW=39.1 KB vs Embedding=1.2 KB
> Dimensionality reduction: 33x
> ```
>
> **Interpretation:** A 4-word sentence creates a 10,000-dimensional vector that's 99.96% zeros—wasteful and uninformative. Embeddings compress this to 300 dense dimensions that actually encode meaning. For a corpus of 1 million documents, that's 39 GB vs 1.2 GB.
>
> *Source: `slide_computations/module8_examples.py` - `demo_bow_sparsity()`*

### TF-IDF

**Improvement**: Weight words by importance.

$$\text{TF-IDF} = \text{TF}(t,d) \times \log\frac{N}{\text{DF}(t)}$$

- **TF** (Term Frequency): How often the word appears in this document
- **IDF** (Inverse Document Frequency): How rare the word is across all documents

Common words like "the" and "is" → low weight
Distinctive words → high weight

**Why the log in IDF?**

1. **Dampening effect**: Without log, a word appearing in 1 vs 1,000 documents would have a 1,000x difference. Log compresses this to about 3x.

2. **Prevents domination**: Extremely rare words would otherwise overwhelm everything else.

Think of it: the difference between appearing in 1 vs 10 documents is more meaningful than 10,000 vs 10,010. The log captures this diminishing-returns intuition.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus)
```

> **Numerical Example: TF-IDF Calculation by Hand**
>
> ```python
> import numpy as np
>
> # Small corpus
> corpus = [
>     "the cat sat on the mat",
>     "the dog ran in the park",
>     "the cat chased the dog",
>     "bankruptcy filing announced today",
> ]
> n_docs = 4
>
> # Analyze "the" vs "bankruptcy" in Doc 1
> doc1_words = corpus[0].split()
> doc1_len = len(doc1_words)  # 6 words
>
> # "the": appears 2x in doc1, in 3/4 docs
> tf_the = 2 / 6  # 0.333
> idf_the = np.log(4 / 3) + 1  # 1.288
> tfidf_the = tf_the * idf_the
> print(f"'the':        TF={tf_the:.3f}, IDF={idf_the:.3f}, TF-IDF={tfidf_the:.3f}")
>
> # "bankruptcy": appears 0x in doc1, in 1/4 docs
> tf_bank = 0 / 6  # 0.000
> idf_bank = np.log(4 / 1) + 1  # 2.386
> tfidf_bank = tf_bank * idf_bank
> print(f"'bankruptcy': TF={tf_bank:.3f}, IDF={idf_bank:.3f}, TF-IDF={tfidf_bank:.3f}")
> ```
>
> **Output:**
> ```
> 'the':        TF=0.333, IDF=1.288, TF-IDF=0.429
> 'bankruptcy': TF=0.000, IDF=2.386, TF-IDF=0.000
> ```
>
> **Interpretation:** Even though "the" appears twice in Doc 1, its TF-IDF is low because it appears in almost every document (low IDF). "Bankruptcy" has high IDF (rare word) but zero TF-IDF in Doc 1 because it doesn't appear there. TF-IDF rewards words that are both frequent in a document AND rare across the corpus.
>
> *Source: `slide_computations/module8_examples.py` - `demo_tfidf_by_hand()`*

### Word Embeddings

**The breakthrough**: Learn dense vectors where similar words are close.

**Word2Vec (2013)**: Train a neural network on word prediction.
- **Skip-gram**: Given a word, predict its context words
- **CBOW**: Given context words, predict the target word
- Result: 100-300 dimensional vectors per word

**The key insight**: The embedding layer weights ARE the word vectors. Words appearing in similar contexts get similar embeddings.

**Famous example:**

$$king - man + woman \approx queen$$

```python
from gensim.models import Word2Vec

model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)
model.wv.most_similar(positive=['king', 'woman'], negative=['man'])
```

This works because the embedding captures semantic relationships! "King" and "queen" differ in the same way that "man" and "woman" differ.

**Think of embeddings as neighborhoods**: In the embedding space, similar words are neighbors. The "royalty neighborhood" contains king, queen, prince, throne. The "food neighborhood" contains apple, banana, pizza. Words can belong to multiple neighborhoods—"apple" is near both "banana" (fruit) and "iPhone" (company). When you subtract "man" from "king," you're finding the direction from the "male" neighborhood to... somewhere. Adding "woman" then moves in the "female" direction. You end up in the same relative position as queen.

**How Word2Vec learns relationships**: Word2Vec never sees labeled examples of gender or royalty—these emerge from the distributional hypothesis (words in similar contexts have similar meanings). The model sees "king" near "throne," "crown," "ruled"; so does "queen." To minimize prediction error, the embedding must encode that "king → queen" is the same direction as "man → woman." This emergent structure falls out naturally from simple prediction tasks on large corpora.

> **Numerical Example: Embedding Similarity**
>
> ```python
> import numpy as np
>
> # Simulated word embeddings (50 dimensions, normalized)
> # Constructed so king-man+woman ≈ queen
> np.random.seed(42)
>
> def cosine_sim(a, b):
>     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
>
> # Create embeddings with semantic structure
> base = np.random.randn(50)
> gender_dir = np.random.randn(50) * 0.5
> royalty_dir = np.random.randn(50) * 0.5
>
> embeddings = {
>     "man": base,
>     "woman": base + gender_dir,
>     "king": base + royalty_dir,
>     "queen": base + gender_dir + royalty_dir,
>     "banana": np.random.randn(50),
> }
>
> # Cosine similarities
> print("Cosine similarities:")
> print(f"  king ↔ queen:  {cosine_sim(embeddings['king'], embeddings['queen']):+.3f}")
> print(f"  king ↔ man:    {cosine_sim(embeddings['king'], embeddings['man']):+.3f}")
> print(f"  king ↔ banana: {cosine_sim(embeddings['king'], embeddings['banana']):+.3f}")
>
> # Analogy: king - man + woman = ?
> result = embeddings["king"] - embeddings["man"] + embeddings["woman"]
> print(f"\nking - man + woman closest to:")
> for word, emb in embeddings.items():
>     print(f"  {word}: {cosine_sim(result, emb):+.3f}")
> ```
>
> **Output:**
> ```
> Cosine similarities:
>   king ↔ queen:  +0.915
>   king ↔ man:    +0.860
>   king ↔ banana: +0.233
>
> king - man + woman closest to:
>   man: +0.765
>   woman: +0.863
>   king: +0.920
>   queen: +1.000 ← closest!
>   banana: +0.228
> ```
>
> **Interpretation:** Similar words (king/queen) have high cosine similarity (~0.9), while unrelated words (king/banana) have low similarity (~0.2). The analogy works because vector arithmetic preserves the learned relationships: subtracting "man" and adding "woman" moves in the gender direction, landing closest to "queen."
>
> *Source: `slide_computations/module8_examples.py` - `demo_embedding_similarity()`*

### Why Context Matters

Word embeddings are powerful, but they miss context:

**Word order:**
- "Nick ate the pizza" vs "The pizza ate Nick"
- Same words, completely different meaning

**Negation:**
- "The movie was good" vs "The movie was not good"
- BoW and simple embeddings can't distinguish these

**Reference:**
- "The dog didn't cross the road because *it* was tired"
- "The dog didn't cross the road because *it* was wide"
- What does "it" refer to? Depends on context!

**Key insight**: We need models that understand sequences and context.

### Common Misconceptions

| Misconception | Reality |
|--------------|---------|
| "Word embeddings understand meaning" | Embeddings capture statistical patterns, not true understanding |
| "Pre-trained embeddings work for any domain" | Domain-specific training often helps (medical, legal) |
| "More dimensions = better embeddings" | Diminishing returns; 100-300 usually sufficient |

---

## 8.2 Recurrent Neural Networks

### RNN Architecture

**Problem**: Standard neural networks can't handle variable-length sequences or remember previous inputs.

**Solution**: Process sequences one element at a time, maintaining memory.

$$h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b)$$

![RNN Unrolled](../assets/module8/rnn_unrolled.png)

**Reading the diagram**: This shows an RNN "unrolled" through time—the same network repeated at each timestep. Reading left to right: blue circles (x1, x2, x3, x4) are inputs at each timestep (e.g., word embeddings). Each input feeds into a purple hidden state box (h1, h2, h3, h4), which also receives information from the previous hidden state via the orange arrows. Orange outputs (y1, y2, y3, y4) can be produced at each step. The key insight: h2 contains information from both x2 AND x1 (via h1). By h4, the hidden state has seen the entire sequence—but early information may be degraded after passing through multiple transformations.

The **hidden state** $h$ carries information through time.

**Think of it like passing notes in class**: Each student (timestep) receives a note from the previous student, reads the new information (input), writes a combined summary, and passes it forward. By the end of the row, the final note contains a compressed summary of everything—but details from early students may be garbled or lost. This is both the power and limitation of RNNs: the hidden state must compress all history into a fixed-size vector.

**Why tanh?**
1. **Output range [-1, 1]**: Can represent "opposite" concepts
2. **Zero-centered**: Helps gradients flow in both directions
3. **Stronger gradients**: Maximum gradient is 1 (vs 0.25 for sigmoid)
4. **Bounded**: Prevents hidden states from exploding

> **Numerical Example: RNN Hidden State Evolution**
>
> ```python
> import numpy as np
>
> np.random.seed(42)
>
> # Simple RNN: h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1})
> input_dim, hidden_dim = 4, 3
> W_xh = np.random.randn(hidden_dim, input_dim) * 0.5
> W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.5
>
> # Word embeddings for "I love ML"
> words = ["I", "love", "ML"]
> embeddings = {
>     "I": np.array([0.2, -0.1, 0.3, 0.1]),
>     "love": np.array([0.8, 0.5, -0.2, 0.3]),
>     "ML": np.array([0.1, 0.4, 0.6, -0.1]),
> }
>
> h = np.zeros(hidden_dim)  # Initial hidden state
>
> for t, word in enumerate(words):
>     x = embeddings[word]
>     h_new = np.tanh(W_xh @ x + W_hh @ h)
>     print(f"t={t+1} '{word}': h = [{', '.join(f'{v:+.2f}' for v in h_new)}]")
>     h = h_new
>
> print(f"\nFinal h encodes: 'I' → 'love' → 'ML'")
> ```
>
> **Output:**
> ```
> t=1 'I':    h = [+0.23, +0.26, -0.17]
> t=2 'love': h = [+0.25, -0.39, -0.45]
> t=3 'ML':   h = [+0.72, +0.41, -0.19]
>
> Final h encodes: 'I' → 'love' → 'ML'
> ```
>
> **Interpretation:** Each hidden state combines the current input with the previous hidden state. By t=3, h₃ contains information from all three words—but compressed into just 3 numbers. The same weights (W_xh, W_hh) are used at every timestep, so the RNN learns patterns that generalize across positions.
>
> *Source: `slide_computations/module8_examples.py` - `demo_rnn_hidden_state()`*

### The Vanishing Gradient Problem

**The challenge**: Gradients shrink exponentially through timesteps.

If you're processing a 100-word sentence, gradients from word 100 need to flow back to word 1. But multiplied through 100 steps, they become tiny.

**The multiplicative decay problem**: If each backpropagation step multiplies the gradient by 0.9 (a reasonable value for tanh derivatives), after 100 steps you have 0.9¹⁰⁰ ≈ 0.00003. The gradient has shrunk to 0.003% of its original size! Information from word 1 effectively has no influence on learning by the time the gradient reaches it.

**Result**: The RNN "forgets" early parts of long sequences.

> **Numerical Example: Vanishing Gradient**
>
> ```python
> import numpy as np
>
> # Gradient multiplied at each timestep by factor < 1
> factor = 0.9  # Typical tanh derivative average
>
> print("Gradient decay through sequence:")
> print(f"{'Timesteps':>12} {'Remaining Gradient':>20}")
> print("-" * 35)
>
> for t in [1, 10, 25, 50, 100]:
>     remaining = factor ** t
>     print(f"{t:>12} {remaining:>20.10f}")
>
> # What this means for learning
> print(f"\nAfter 100 timesteps:")
> print(f"  Gradient reduced to: {0.9**100:.6f} = {0.9**100*100:.4f}%")
> print(f"  Information from word 1 has almost no influence on learning")
> ```
>
> **Output:**
> ```
> Gradient decay through sequence:
>    Timesteps    Remaining Gradient
> -----------------------------------
>            1           0.9000000000
>           10           0.3486784401
>           25           0.0717897988
>           50           0.0051537752
>          100           0.0000265614
>
> After 100 timesteps:
>   Gradient reduced to: 0.000027 = 0.0027%
>   Information from word 1 has almost no influence on learning
> ```
>
> **Interpretation:** With each timestep, gradients are multiplied by ~0.9. After just 50 steps, only 0.5% of the gradient remains. After 100 steps, the gradient is 0.003% of its original value—essentially zero. This is why standard RNNs cannot learn long-range dependencies: the error signal from late words never reaches early words during training.
>
> *Source: `slide_computations/module8_examples.py` - `demo_vanishing_gradient()`*

### LSTM: Long Short-Term Memory

**Solution**: Gated architecture with explicit memory.

**Three gates:**
1. **Forget gate**: What to remove from memory
2. **Input gate**: What new information to add
3. **Output gate**: What to output

**Cell state**: A highway for information to flow unchanged through time.

**Think of it like a secretary managing a filing cabinet**: The filing cabinet (cell state) holds long-term memory. When new information arrives, the secretary decides: (1) What old files to shred (forget gate), (2) What new information to file away (input gate), and (3) What to pull out for the current task (output gate). Unlike the "passing notes" RNN where everything gets rewritten each step, the filing cabinet preserves information until explicitly discarded.

The gates learn when to keep information and when to forget it.

```python
lstm = nn.LSTM(
    input_size=100,
    hidden_size=256,
    num_layers=2,
    batch_first=True,
    bidirectional=True
)
```

**Connection to attention**: LSTM gates pioneered the idea of selective information access. Attention generalizes this—instead of a single memory cell, attention lets the model look back at any previous position.

### GRU: Gated Recurrent Unit

**Simplified LSTM** with fewer parameters.

**Two gates:**
1. **Reset gate**: How much past to forget
2. **Update gate**: How much to update the hidden state

Often performs similarly to LSTM but trains faster.

### RNN Limitations

1. **Sequential processing**: Can't parallelize—each step depends on the previous
2. **Long-range dependencies**: Still struggle with very long sequences
3. **Fixed representation**: A single hidden vector must capture everything

**These limitations motivated transformers.**

**Why RNNs dominated before transformers**: They were the best available option. Before RNNs: n-gram models (limited context, exponential parameters) and HMMs (restrictive assumptions). LSTMs/GRUs mitigated vanishing gradients; attention mechanisms (2014-2015) addressed the fixed-representation bottleneck. The 2017 transformer paper showed attention alone was sufficient, but required significant innovations (positional encoding, Q/K/V formulation) plus computational resources. Progress looks obvious in retrospect.

---

## 8.3 Transformers

### "Attention Is All You Need" (2017)

This paper changed everything.

**The key insight**: Replace recurrence with attention.

**Benefits:**
- **Parallel processing**: Process all tokens simultaneously
- **Direct connections**: Any position can attend to any other
- **Better long-range dependencies**: No vanishing gradient through 100 steps

### Self-Attention

**Core idea**: Each word looks at all other words to understand context.

**Query, Key, Value:**
- **Query (Q)**: What am I looking for?
- **Key (K)**: What do I contain?
- **Value (V)**: What information do I provide?

**Think of it like searching a library**: You have a question (Query). Each book has a title and keywords (Keys) that describe what it contains. The book's actual content is the Value. You compare your question against all book titles (Q·K), find the most relevant matches (softmax), then read and combine information from those books (weighted sum of Values). A word asking "what does 'it' refer to?" searches all other words' keys, finds "cat" is most relevant, and copies cat's information.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Intuition:**
1. Compute similarity between query and all keys
2. Normalize with softmax → attention weights
3. Weighted sum of values

**Example**: "The cat sat on the mat because **it** was tired"

When processing "it":
- Compute similarity with all words
- "it" should attend most strongly to "cat"
- Copy information from "cat" to understand what "it" refers to

**How attention learns coreference**: Entirely through training—nothing programmed in. "It was tired" makes sense if "it" attends to "cat" (animals get tired), not "mat." The Q/K/V projection matrices adjust so "it" and "cat" have high dot product. Different heads specialize: one for coreference, another for syntax, another for local context. The model discovers these patterns; engineers didn't program them.

> **Numerical Example: Self-Attention Step by Step**
>
> ```python
> import numpy as np
>
> np.random.seed(42)
>
> # 3-word sentence, 4-dim embeddings, 3-dim Q/K/V
> words = ["The", "cat", "sat"]
> X = np.array([[0.1, 0.2, 0.3, 0.4],    # The
>               [0.5, 0.6, -0.2, 0.1],   # cat
>               [0.2, -0.1, 0.4, 0.3]])  # sat
>
> W_Q = np.random.randn(4, 3) * 0.5
> W_K = np.random.randn(4, 3) * 0.5
>
> Q = X @ W_Q  # Queries
> K = X @ W_K  # Keys
>
> # Attention scores: Q @ K^T
> scores = Q @ K.T
> scaled = scores / np.sqrt(3)  # Scale by √d_k
>
> # Softmax
> def softmax(x):
>     exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
>     return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
>
> attention = softmax(scaled)
>
> print("Attention weights (each row = what that word attends to):")
> print(f"       {'The':>6} {'cat':>6} {'sat':>6}")
> for i, word in enumerate(words):
>     print(f"{word:>6} {attention[i,0]:>6.2f} {attention[i,1]:>6.2f} {attention[i,2]:>6.2f}")
> ```
>
> **Output:**
> ```
> Attention weights (each row = what that word attends to):
>           The    cat    sat
>    The   0.32   0.35   0.33
>    cat   0.33   0.34   0.33
>    sat   0.33   0.34   0.33
> ```
>
> **Interpretation:** Each row shows where that word "looks." With random weights, attention is nearly uniform. After training, you'd see patterns like "sat" attending strongly to "cat" (subject-verb relationship) or pronouns attending to their referents. The softmax ensures weights sum to 1, creating a weighted average of all positions.
>
> *Source: `slide_computations/module8_examples.py` - `demo_self_attention()`*

### Why Scale by √d_k?

Dot products grow with dimension. If d_k is large, dot products can be very large, pushing softmax into saturation (all attention on one token). Scaling keeps variance roughly constant.

### Multi-Head Attention

**Why multiple heads?** Different heads can attend to different things.

- One head might focus on syntax (subject-verb agreement)
- Another might focus on semantics (what "it" refers to)
- Another might focus on nearby context

```python
multihead_attn = nn.MultiheadAttention(
    embed_dim=512,
    num_heads=8
)
```

Eight heads, each with 64 dimensions, capturing different relationships.

### Positional Encoding

**Problem**: Attention is permutation-invariant. It doesn't know word order!

**Solution**: Add position information to embeddings.

$$PE_{pos,2i} = \sin(pos / 10000^{2i/d})$$

$$PE_{pos,2i+1} = \cos(pos / 10000^{2i/d})$$

Where:
- $pos$ = position in sequence (0, 1, 2, ...)
- $d$ = embedding dimension
- $i$ = dimension index, ranging from 0 to $d/2 - 1$

**Think of it like clock hands**: Low dimensions are like a second hand—they cycle rapidly (short wavelength), changing noticeably between adjacent positions. High dimensions are like an hour hand—they cycle slowly (long wavelength), barely changing between nearby positions but clearly different across the sequence. Together, they create a unique "fingerprint" for each position. Just as you can tell time by combining all hands, the model can determine position from the combined pattern.

> **Numerical Example: Positional Encoding Patterns**
>
> ```python
> import numpy as np
>
> d_model = 8
> max_pos = 6
>
> # Compute positional encodings
> PE = np.zeros((max_pos, d_model))
> for pos in range(max_pos):
>     for i in range(d_model // 2):
>         denom = 10000 ** (2 * i / d_model)
>         PE[pos, 2*i] = np.sin(pos / denom)
>         PE[pos, 2*i + 1] = np.cos(pos / denom)
>
> print("Positional encodings (dims 0-3):")
> print(f"Pos  {'dim0':>7} {'dim1':>7} {'dim2':>7} {'dim3':>7}")
> for pos in range(max_pos):
>     print(f"{pos:>3}  {PE[pos,0]:+.3f}  {PE[pos,1]:+.3f}  {PE[pos,2]:+.3f}  {PE[pos,3]:+.3f}")
>
> # Wavelengths
> print(f"\nWavelength (positions per cycle):")
> for i in range(d_model // 2):
>     wl = 2 * np.pi * (10000 ** (2*i/d_model))
>     print(f"  dims {2*i},{2*i+1}: {wl:.1f} positions")
> ```
>
> **Output:**
> ```
> Positional encodings (dims 0-3):
> Pos     dim0    dim1    dim2    dim3
>   0  +0.000  +1.000  +0.000  +1.000
>   1  +0.841  +0.540  +0.100  +0.995
>   2  +0.909  -0.416  +0.199  +0.980
>   3  +0.141  -0.990  +0.296  +0.955
>   4  -0.757  -0.654  +0.389  +0.921
>   5  -0.959  +0.284  +0.479  +0.878
>
> Wavelength (positions per cycle):
>   dims 0,1: 6.3 positions
>   dims 2,3: 62.8 positions
>   dims 4,5: 628.3 positions
>   dims 6,7: 6283.2 positions
> ```
>
> **Interpretation:** Dims 0,1 complete a full cycle every ~6 positions (fast "second hand"). Dims 6,7 take ~6,000 positions to cycle (slow "hour hand"). Each position gets a unique combination of values. The model learns to use these patterns to determine both absolute position and relative distances between tokens.
>
> *Source: `slide_computations/module8_examples.py` - `demo_positional_encoding()`*

Different frequencies let the model learn to attend to relative positions.

### Encoder vs Decoder

**Encoder (BERT-style):**
- Processes entire sequence at once
- Bidirectional context (see past and future)
- Good for understanding and classification

**Decoder (GPT-style):**
- Generates sequence left-to-right
- Causal masking (can only see past)
- Good for text generation

**Encoder-Decoder (T5):**
- Encoder processes input
- Decoder generates output
- Good for translation, summarization

### Common Misconceptions

| Misconception | Reality |
|--------------|---------|
| "Transformers understand language" | They learn statistical patterns, not true understanding |
| "Attention = interpretability" | Attention weights don't always align with human intuition |
| "Bigger models are always better" | Diminishing returns; efficiency matters |

---

## 8.4 Foundation Models

### Pre-training → Fine-tuning

**Pre-training**: Train on massive text (expensive!)
- Billions of words
- Millions of dollars in compute
- Done once by big labs

**Fine-tuning**: Adapt to your task (cheap!)
- Your data + pre-trained model
- Hours, not weeks

**Zero-shot**: Use directly with prompts
- No training needed
- Just ask the model

### BERT

**Bidirectional Encoder Representations from Transformers**

**Pre-training:**
- Masked Language Modeling: Predict masked words from context
- Next Sentence Prediction: Does sentence B follow sentence A?

**Use cases:**
- Text classification
- Named entity recognition
- Question answering
- Semantic similarity

```python
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

inputs = tokenizer(
    "This movie was great!",
    return_tensors="pt",
    padding=True,
    truncation=True
)
outputs = model(**inputs)
```

### GPT Family

**Generative Pre-trained Transformer**

**Architecture**: Decoder-only (autoregressive)

**Capabilities:**
- Text generation
- Zero/few-shot learning
- Instruction following (ChatGPT)

**Scale evolution:**
- GPT (2018): 117M parameters
- GPT-2 (2019): 1.5B parameters
- GPT-3 (2020): 175B parameters
- GPT-4 (2023): Multimodal, even larger

### BERT vs GPT

| Aspect | BERT | GPT |
|--------|------|-----|
| Architecture | Encoder | Decoder |
| Context | Bidirectional | Left-to-right |
| Best for | Understanding | Generation |
| Training | Masked LM | Next token prediction |

**When to use which?**

Use BERT for classification, NER, and understanding tasks—especially with labeled training data.

Use GPT for generation tasks, or when you want to leverage prompting without training data.

**A practical decision framework**:
| Your Situation | Recommendation |
|----------------|----------------|
| <100 labeled examples | GPT zero/few-shot |
| 100-1,000 labeled examples | Try both, compare |
| >1,000 labeled examples | Fine-tune BERT (likely wins) |
| Need real-time inference | BERT (faster, cheaper) |
| Need to generate text | GPT |
| Domain-specific vocabulary | Fine-tune either on domain text |

**Why fine-tune BERT vs. zero-shot GPT?** (1) Task-specific performance: fine-tuned BERT typically achieves higher accuracy with sufficient training data. (2) Cost/latency: BERT-base (110M params) is orders of magnitude cheaper than GPT-4 (1T+ params). (3) Consistency: fine-tuned models are deterministic; GPT varies with temperature and prompts. (4) Domain adaptation and data privacy (local training vs. API calls). Use both strategically: GPT for exploration, fine-tuned BERT for production systems.

> **Numerical Example: BERT vs GPT Scale Comparison**
>
> ```python
> models = {
>     "BERT-base":  {"params": 110_000_000,  "layers": 12, "hidden": 768},
>     "BERT-large": {"params": 340_000_000,  "layers": 24, "hidden": 1024},
>     "GPT-2":      {"params": 1_500_000_000, "layers": 48, "hidden": 1600},
>     "GPT-3":      {"params": 175_000_000_000, "layers": 96, "hidden": 12288},
> }
>
> print(f"{'Model':<12} {'Parameters':>12} {'Layers':>8} {'Hidden':>8}")
> print("-" * 45)
> for name, specs in models.items():
>     p = specs['params']
>     p_str = f"{p/1e9:.1f}B" if p >= 1e9 else f"{p/1e6:.0f}M"
>     print(f"{name:<12} {p_str:>12} {specs['layers']:>8} {specs['hidden']:>8}")
>
> # Relative cost (rough)
> bert_base = 110e6
> print(f"\nRelative inference cost (vs BERT-base):")
> for name, specs in models.items():
>     ratio = specs['params'] / bert_base
>     print(f"  {name}: ~{ratio:.0f}x")
> ```
>
> **Output:**
> ```
> Model         Parameters   Layers   Hidden
> ---------------------------------------------
> BERT-base          110M       12      768
> BERT-large         340M       24     1024
> GPT-2              1.5B       48     1600
> GPT-3            175.0B       96    12288
>
> Relative inference cost (vs BERT-base):
>   BERT-base: ~1x
>   BERT-large: ~3x
>   GPT-2: ~14x
>   GPT-3: ~1591x
> ```
>
> **Interpretation:** GPT-3 is ~1,600x more expensive to run than BERT-base. For a classification task processing 1 million documents, BERT-base might cost $10 while GPT-3 costs $16,000. This is why production systems often use fine-tuned BERT for tasks where it performs well—the cost difference is dramatic.
>
> *Source: `slide_computations/module8_examples.py` - `demo_bert_vs_gpt_scale()`*

### Business Applications

| Application | Model | Example |
|-------------|-------|---------|
| Sentiment Analysis | BERT | Product reviews |
| Chatbot | GPT | Customer support |
| Classification | BERT | Email routing |
| Named Entity Recognition | BERT | Extract entities |
| Text Generation | GPT | Marketing copy |
| Summarization | T5, BART | Meeting notes |

---

## 8.5 Beyond Text

### Vision Transformers (ViT)

- Split images into patches
- Treat patches as "tokens"
- Apply transformer encoder
- State-of-the-art on many vision benchmarks

### Audio Processing

- **Whisper**: Speech recognition
- **wav2vec**: Audio embeddings

### Multimodal Models

- **CLIP**: Connect images and text
- **DALL-E**: Generate images from text
- **GPT-4V**: Vision + language

**Key insight**: Transformer architecture is general-purpose, not just for text.

---

## Reflection Questions

1. Why does 'king - man + woman ≈ queen' work with word embeddings?

2. A BoW model can't tell 'dog bites man' from 'man bites dog'. Why not? What's needed to fix this?

3. You're building a document search engine. Would you use BoW, TF-IDF, or embeddings? Why?

4. Why can't a standard feedforward network process variable-length text?

5. An LSTM processes a 100-word sentence. How does information from word 1 reach the output?

6. Why is self-attention more parallelizable than RNNs?

7. In "The animal didn't cross the road because it was tired", what should 'it' attend to?

8. Why does BERT use bidirectional attention while GPT uses causal attention?

9. When would you fine-tune BERT vs use GPT with prompting?

---

## Practice Problems

1. For a vocabulary of 10,000 words and a 5-word document, what's the dimensionality of BoW vs a 300-dim embedding?

2. Calculate TF-IDF for a word appearing 3 times in a document, when it appears in 100 of 10,000 documents.

3. Explain why RNNs suffer from vanishing gradients but LSTMs partially solve this.

4. Given Q, K, V matrices, trace through the self-attention computation.

5. A company wants to classify support tickets. Recommend BERT vs GPT and justify.

---

## Chapter Summary

**Six key takeaways from Module 8:**

1. **Text representation evolves**: BoW → TF-IDF → embeddings → contextual embeddings

2. **RNNs** process sequences but struggle with long-range dependencies

3. **Transformers** use attention for parallel, effective processing

4. **Self-attention** lets each token consider all others

5. **BERT** for understanding, **GPT** for generation

6. **Transfer learning** makes NLP accessible

---

## What's Next

In Module 9, we tackle **Model Interpretability**:
- Why do models make decisions?
- SHAP values
- Attention visualization
- Building trust in ML systems

We'll use attention from transformers to understand what NLP models focus on!
