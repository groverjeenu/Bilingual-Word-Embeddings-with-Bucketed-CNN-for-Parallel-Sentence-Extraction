# Bilingual Word Embeddings with Bucketed CNN for Parallel Sentence Extraction

A Tensorflow Flow implementation of our recent paper in ACL 2017(Student track) ([Bilingual Word Embeddings with Bucketed CNN for Parallel Sentence Extraction.](http://www.aclweb.org/anthology/P17-3003))

Two sentences are said to be aligned or semantically similar if they convey the same semantics in both the languages. Our code makes use of the Bilingual Word Embeddings for capturing the smetic s related of two words across languages.
A similarity matrix is constructed between the words of two sentences, which is dynamically pooled to a fixed size dimension 'dim' for classification tasks.
We split the data into different bucket sizes as one fixed size representation would not work effectively for all sentence-pair sizes. Separate CNN's were trained on each data split.

![Model Overview](assets/Model.png)




## License

MIT

