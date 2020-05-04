## Updates

#### 2020-4-20 14:13:48

* Start the first project： Q&A Summarization and Inference
* Process data and train word vector using word2vec(skip-gram)



#### 2020-5-4 09:26:38

##### Code

- Re-name folder structure name
- Add config folder to manage data path
- Build embedding matrix with three methods: Word2Vec, FastText and Tencent open source 



##### Details

- Special words process before Jieba segmentation

  1. Remove all special symbols, just keep character and number?

  2. Keep necessary punctuations like，。?

  3. Need pay attention to special blank/symbols crawled from Web

     ```py
     words = set(['\u3000', '\u81a8', '\u5316', '\u98df', '\u54c1', '\xa0', '\u00a0', '\u2002', '\u2003'
     ```

  

- OOV of embedding:

  1. Three methods comparison

     ```
     Found 85993/120320 words in: Tencent_AILab_ChineseEmbedding.txt
     Found 32427/120320 words in: word2vector.model
     Found 120320/120320 words in: fasttest.model
     ```

     

  