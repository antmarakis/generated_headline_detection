## Identifying Automatically Generated Headlines using Transformers

Code for [NLP4IF 2021](http://www.netcopia.net/nlp4if/) workshop paper.

### How to replicate results?

Step 1: Download Data.
  * [Link to generated headlines data (attack).](https://drive.google.com/file/d/1pGNqrQmNxIlKDQYux9mCip-H-bSsreSY/view?usp=sharing)
  * [Link to generated headlines data (defense).](https://drive.google.com/file/d/1y_zFKSpJ2CTRnSoUq7cz3q4Ir-bERKVl/view?usp=sharing)
  * For the real headlines, go to [Million Headlines](https://www.kaggle.com/therohk/million-headlines), or contact me for the final dataset (due to copyright concerns I cannot share the data myself publicly).

Step 2: Pretrain two LMs using the [HuggingFace library](https://github.com/huggingface/transformers/tree/master/examples/language-modeling). Pretrain one LM on the "attack" data and another on the "defense".

Step 3: Generate headlines for attack and defense (if you haven't downloaded the ones from Step 1).

Step 4: Merge the real/generated headlines for attack and defense.

Step 5: Finetune your classifiers (using the example scripts provided here, eg. `bert_classify.py`).
