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




#### 2020-5-21 17:47:07

##### Code

- Finish seq2seq baseline; functional run with small sample size can pass
- Full data run and hyper parameters study on going...



#### 2020-5-26 10:54:18

##### Training logs

- Loss converge very slow
  - learning rate from 0.00001 -> 0.001, no big help
  - Special token embedding: change vector values closer to 0, no big help    
  - Embedding model: myself trained W2V -> Tencent model, no big help

- Skip converge issue temporarily, keep going on PGN/Coverage/Transformer

```pytho
# epoch=10
# steps=1300
# bz=64
# lr=0.001
# dec/uec_units = 200
# atten_units = 100
# tecent embedding model

[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
Creating the model ...
Generate embedding from /content/drive/My Drive/kaikeba_nlp2/projects/P01_QA_summarization_inference/data/processed/tencent_embedding.pkl
max_size of vocab was specified as 30000; we now have 30000 words. Stopping reading.
Finished constructing vocabulary 
of 30000 total words. Last word added: 红接
save /content/drive/My Drive/kaikeba_nlp2/projects/P01_QA_summarization_inference/data/processed/tencent_embedding.pkl ok.
Creating the checkpoint manager ...
Initializing from scratch
Start the training ...
max_size of vocab was specified as 30000; we now have 30000 words. Stopping reading.
Finished constructing vocabulary of 30000 total words. Last word added: 红接
True vocab is <projects.P01_QA_summarization_inference.src.utils.batcher.Vocab object at 0x7fe20cba2518>
Creating the batch set ...
Epoch 1 Batch 100 Loss 6.6398
Epoch 1 Batch 200 Loss 6.4416
Epoch 1 Batch 300 Loss 6.0321
Epoch 1 Batch 400 Loss 6.0121
Epoch 1 Batch 500 Loss 5.4782
Epoch 1 Batch 600 Loss 5.5707
Epoch 1 Batch 700 Loss 5.3484
Epoch 1 Batch 800 Loss 5.5706
Epoch 1 Batch 900 Loss 5.2996
Epoch 1 Batch 1000 Loss 5.2655
Epoch 1 Batch 1100 Loss 5.1746
Epoch 1 Batch 1200 Loss 5.0299
Epoch 1 Batch 1300 Loss 5.2135
Saving check point for epoch 1 at /content/drive/My Drive/kaikeba_nlp2/projects/P01_QA_summarization_inference/models/seq2seq/checkpoint/ckpt-1, best loss 5.7393
Epoch 1 Loss 5.7393
Time taken for 1 epoch 759.0938351154327 sec

Epoch 2 Batch 100 Loss 4.9400
Epoch 2 Batch 200 Loss 4.7516
Epoch 2 Batch 300 Loss 4.7206
Epoch 2 Batch 400 Loss 4.8361
Epoch 2 Batch 500 Loss 4.8389
Epoch 2 Batch 600 Loss 4.9038
Epoch 2 Batch 700 Loss 5.0174
Epoch 2 Batch 800 Loss 5.1195
Epoch 2 Batch 900 Loss 5.1211
Epoch 2 Batch 1000 Loss 5.2393
Epoch 2 Batch 1100 Loss 4.9981
Epoch 2 Batch 1200 Loss 4.9336
Epoch 2 Batch 1300 Loss 4.7318
Saving check point for epoch 2 at /content/drive/My Drive/kaikeba_nlp2/projects/P01_QA_summarization_inference/models/seq2seq/checkpoint/ckpt-2, best loss 4.9356
Epoch 2 Loss 4.9356
Time taken for 1 epoch 716.4182484149933 sec

Epoch 3 Batch 100 Loss 4.4180
Epoch 3 Batch 200 Loss 4.8400
Epoch 3 Batch 300 Loss 4.8288
Epoch 3 Batch 400 Loss 4.7261
Epoch 3 Batch 500 Loss 4.6006
Epoch 3 Batch 600 Loss 4.5325
Epoch 3 Batch 700 Loss 4.7580
Epoch 3 Batch 800 Loss 4.6844
Epoch 3 Batch 900 Loss 4.9736
Epoch 3 Batch 1000 Loss 4.8765
Epoch 3 Batch 1100 Loss 4.8281
Epoch 3 Batch 1200 Loss 4.6831
Epoch 3 Batch 1300 Loss 4.8777
Saving check point for epoch 3 at /content/drive/My Drive/kaikeba_nlp2/projects/P01_QA_summarization_inference/models/seq2seq/checkpoint/ckpt-3, best loss 4.7277
Epoch 3 Loss 4.7277
Time taken for 1 epoch 695.4865171909332 sec

Epoch 4 Batch 100 Loss 4.5981
Epoch 4 Batch 200 Loss 4.5232
Epoch 4 Batch 300 Loss 4.5103
Epoch 4 Batch 400 Loss 4.5150
Epoch 4 Batch 500 Loss 4.7275
Epoch 4 Batch 600 Loss 4.7273
Epoch 4 Batch 700 Loss 4.6424
Epoch 4 Batch 800 Loss 4.9208
Epoch 4 Batch 900 Loss 4.6247
Epoch 4 Batch 1000 Loss 4.5958
Epoch 4 Batch 1100 Loss 4.7401
Epoch 4 Batch 1200 Loss 4.5026
Epoch 4 Batch 1300 Loss 4.5655
Saving check point for epoch 4 at /content/drive/My Drive/kaikeba_nlp2/projects/P01_QA_summarization_inference/models/seq2seq/checkpoint/ckpt-4, best loss 4.6008
Epoch 4 Loss 4.6008
Time taken for 1 epoch 689.9416298866272 sec

Epoch 5 Batch 100 Loss 4.2893
Epoch 5 Batch 200 Loss 4.4055
Epoch 5 Batch 300 Loss 4.5711
Epoch 5 Batch 400 Loss 4.6279
Epoch 5 Batch 500 Loss 4.5861
Epoch 5 Batch 600 Loss 4.6034
Epoch 5 Batch 700 Loss 4.7254
Epoch 5 Batch 800 Loss 4.6000
Epoch 5 Batch 900 Loss 4.5777
Epoch 5 Batch 1000 Loss 4.8938
Epoch 5 Batch 1100 Loss 4.6214
Epoch 5 Batch 1200 Loss 4.4008
Epoch 5 Batch 1300 Loss 4.6166
Saving check point for epoch 5 at /content/drive/My Drive/kaikeba_nlp2/projects/P01_QA_summarization_inference/models/seq2seq/checkpoint/ckpt-5, best loss 4.5097
Epoch 5 Loss 4.5097
Time taken for 1 epoch 686.2486438751221 sec

Epoch 6 Batch 100 Loss 4.3993
Epoch 6 Batch 200 Loss 4.3771
Epoch 6 Batch 300 Loss 4.1635
Epoch 6 Batch 400 Loss 4.4842
Epoch 6 Batch 500 Loss 4.4293
Epoch 6 Batch 600 Loss 4.4596
Epoch 6 Batch 700 Loss 4.5603
Epoch 6 Batch 800 Loss 4.6738
Epoch 6 Batch 900 Loss 4.5537
Epoch 6 Batch 1000 Loss 4.4548
Epoch 6 Batch 1100 Loss 4.3690
Epoch 6 Batch 1200 Loss 4.0123
Epoch 6 Batch 1300 Loss 4.3382
Saving check point for epoch 6 at /content/drive/My Drive/kaikeba_nlp2/projects/P01_QA_summarization_inference/models/seq2seq/checkpoint/ckpt-6, best loss 4.4340
Epoch 6 Loss 4.4340
Time taken for 1 epoch 678.5873117446899 sec

Epoch 7 Batch 100 Loss 4.1119
Epoch 7 Batch 200 Loss 3.9650
Epoch 7 Batch 300 Loss 4.2260
Epoch 7 Batch 400 Loss 4.4302
Epoch 7 Batch 500 Loss 4.4960
Epoch 7 Batch 600 Loss 4.4529
Epoch 7 Batch 700 Loss 4.4447
Epoch 7 Batch 800 Loss 4.5514
Epoch 7 Batch 900 Loss 4.3193
Epoch 7 Batch 1000 Loss 4.5174
Epoch 7 Batch 1100 Loss 4.4220
Epoch 7 Batch 1200 Loss 4.4422
Epoch 7 Batch 1300 Loss 4.5152
Saving check point for epoch 7 at /content/drive/My Drive/kaikeba_nlp2/projects/P01_QA_summarization_inference/models/seq2seq/checkpoint/ckpt-7, best loss 4.3650
Epoch 7 Loss 4.3650
Time taken for 1 epoch 682.2583892345428 sec

Epoch 8 Batch 100 Loss 4.2576
Epoch 8 Batch 200 Loss 4.1505
Epoch 8 Batch 300 Loss 4.0799
Epoch 8 Batch 400 Loss 4.2695
Epoch 8 Batch 500 Loss 4.4913
Epoch 8 Batch 600 Loss 4.1179
Epoch 8 Batch 700 Loss 4.4346
Epoch 8 Batch 800 Loss 4.4703
Epoch 8 Batch 900 Loss 4.2824
Epoch 8 Batch 1000 Loss 4.4426
Epoch 8 Batch 1100 Loss 4.4911
Epoch 8 Batch 1200 Loss 4.2380
Epoch 8 Batch 1300 Loss 4.4440
Saving check point for epoch 8 at /content/drive/My Drive/kaikeba_nlp2/projects/P01_QA_summarization_inference/models/seq2seq/checkpoint/ckpt-8, best loss 4.3046
Epoch 8 Loss 4.3046
Time taken for 1 epoch 678.5520474910736 sec

Epoch 9 Batch 100 Loss 3.9721
Epoch 9 Batch 200 Loss 4.1072
Epoch 9 Batch 300 Loss 4.1950
Epoch 9 Batch 400 Loss 3.9984
Epoch 9 Batch 500 Loss 4.4452
Epoch 9 Batch 600 Loss 4.2708
Epoch 9 Batch 700 Loss 4.2726
Epoch 9 Batch 800 Loss 4.4413
Epoch 9 Batch 900 Loss 4.2453
Epoch 9 Batch 1000 Loss 4.2084
Epoch 9 Batch 1100 Loss 4.2769
Epoch 9 Batch 1200 Loss 4.1871
Epoch 9 Batch 1300 Loss 4.3709
Saving check point for epoch 9 at /content/drive/My Drive/kaikeba_nlp2/projects/P01_QA_summarization_inference/models/seq2seq/checkpoint/ckpt-9, best loss 4.2508
Epoch 9 Loss 4.2508
Time taken for 1 epoch 677.9140005111694 sec

Epoch 10 Batch 100 Loss 3.9456
Epoch 10 Batch 200 Loss 4.2572
Epoch 10 Batch 300 Loss 4.1509
Epoch 10 Batch 400 Loss 4.5625
Epoch 10 Batch 500 Loss 4.2140
Epoch 10 Batch 600 Loss 4.2718
Epoch 10 Batch 700 Loss 4.2824
Epoch 10 Batch 800 Loss 4.4362
Epoch 10 Batch 900 Loss 4.2505
Epoch 10 Batch 1000 Loss 4.4225
Epoch 10 Batch 1100 Loss 4.3476
Epoch 10 Batch 1200 Loss 4.0995
Epoch 10 Batch 1300 Loss 4.1888
Saving check point for epoch 10 at /content/drive/My Drive/kaikeba_nlp2/projects/P01_QA_summarization_inference/models/seq2seq/checkpoint/ckpt-10, best loss 4.2057
Epoch 10 Loss 4.2057
Time taken for 1 epoch 680.7809522151947 sec
```