# hard-nce-el

## requirements
transformers 3.1.0, pytorch 1.7.1

## scripts

### retrieval
```
python main_retriever.py --model [model saving path]  --data_dir [zeshel data directory] --B 16 --gradient_accumulation_steps 2 --logging_steps 1000 --k 64 --epochs 4 --lr 0.00001 --num_cands 64 --type_cands hard_and_random_negative --cands_ratio 0.5   --gpus 3,4,5,7    --type_model sum_max  --num_mention_vecs 128 --num_entity_vecs 128 -store_en_hiddens --en_hidden_path [the path for saving all the entity embeddings]  --entity_bsz 4096  --mention_bsz 200

```

### save retrieved candidates
```


```


### reranking
```
python main.py --model [model saving path] --data [zeshel data directory] --B 2  --gradient_accumulation_steps 2 --num_workers 2 --warmup_proportion 0.2 --epochs 3  --gpus 5  --lr 2e-5 --cands_dir [candidates file directory]  --eval_method [micro or macro] --type_model full --type_bert base  --inputmark

```

