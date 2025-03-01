# %% [markdown]
# # Latent Dirichlet Allocation
# %%
import numpy as np
from ..playlist.manager import Playlist, read_csv, split_genres
# %% [markdown]
# ## using sklearn

# %%
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer



# %%
new_liked=liked.songs.reset_index()
index_genres=new_liked.dropna(subset='Genres').reset_index(drop=True)
corpus=index_genres["Genres"].values


# %%

# Convert genres into a bag-of-words model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

# Apply LDA
num_topics=25
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(X)


# Display the topics and their most frequent genres
terms = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    print(f"Topic {topic_idx + 1}:")
    print(", ".join([terms[i] for i in topic.argsort()[:-6 - 1:-1]]))

# You can visualize the topic distributions for tracks if needed
topic_dist = lda.transform(X)


# %% [markdown]
# ## using gensim

# %%
import gensim
from gensim.models import LdaModel, CoherenceModel
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess

# Tokenize the documents
corpus = split_genres(index_genres)
# Create a dictionary from the tokenized corpus
dictionary = Dictionary(corpus)

# Create a bag-of-words (BoW) representation of the corpus
bow_corpus = [dictionary.doc2bow(text) for text in corpus]


# %% [markdown]
# ### Train and Calc Coherence
# Now you can calculate the coherence score using Gensim's CoherenceModel

# %%
# 10 models to test
lda_gensims=[0]*10
coherence_model_ldas=[0]*10
coherence_ldas=[0]*10
for i in range(10):
    # Train the LDA model using Gensim
    lda_gensims[i] = LdaModel(corpus=bow_corpus, id2word=dictionary, num_topics=num_topics)
    coherence_model_ldas[i] = CoherenceModel(model=lda_gensims[i], texts=corpus, dictionary=dictionary, coherence='c_v')
    coherence_ldas[i] = coherence_model_ldas[i].get_coherence()
    print(f'Coherence Score: {coherence_ldas[i]}')




# %% [markdown]
# ### Select the best

# %%
m=np.argmax(coherence_ldas)
print("Max i =",np.argmax(coherence_ldas),"; Max Score =",max(coherence_ldas))

lda_gensim=lda_gensims[m]
coherence_model_lda=coherence_model_ldas[m]
coherence_lda=coherence_ldas[m]

# %% [markdown]
# ### Print the top 5 words for each topic
# 

# %%
topics = lda_gensim.show_topics(num_topics=num_topics, formatted=True)
for topic in topics:
    print(topic)

# %% [markdown]
# ### Print topics for each track (document)
# 

# %%
for i, doc_bow in enumerate(bow_corpus):
    print(f"Document {i}:")
    topic_probabilities = lda_gensim.get_document_topics(doc_bow)
    for topic_id, prob in topic_probabilities:
        print(f"  Topic {topic_id}: Probability {prob:.4f}")

# %%
dominant_topics=[]
indexes=[]
for i, doc_bow in enumerate(bow_corpus):
    topic_probabilities = lda_gensim.get_document_topics(doc_bow)
    dominant_topic = max(topic_probabilities, key=lambda x: x[1])  # Select topic with highest probability
    if dominant_topic[1]>.6:
        # print(f"Document {i} -> Dominant Topic: {dominant_topic[0]} with Probability {dominant_topic[1]:.4f}")
        indexes.append(i)
        dominant_topics.append(dominant_topic[0])

# %% [markdown]
# ### Creating topic based playlists

# %%
print(f' {len(dominant_topics)} found out of {len(index_genres)}.')
new_playlist=index_genres.loc[indexes]
new_playlist["Topic"]=dominant_topics
new_playlist[["Track ID","Track Name","Artist Name(s)", "Topic"]]

# %%
t_idx=3
print(topics[t_idx])
topic_playlist=new_playlist[new_playlist["Topic"]==t_idx]
topic_playlist.head()

# %%
# t_idx playlist
topic_playlist.to_csv("out/topic_playlist.csv",sep=';',decimal=',')

# %%
if len("te st".split(" "))>1: 
    print("te st".split(" ")[1])

# %%

idm=Playlist(read_csv('out/nu_disco_idm.csv')) # I must have created this csv somehwere lol

idm.set_clusters(no_of_clusters=2)

idm.process2([0, 1],in_place=True)

fig=idm.plot3d(index_text=True)

print('\n'.join(idm.songs["Track ID"].values))

# %%

