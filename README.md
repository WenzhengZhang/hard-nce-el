# hard-nce-el
This is a pytorch implementation of the paper [Understanding Hard Negatives in Noise Contrastive Estimation](https://arxiv.org/pdf/2104.06245.pdf) [1].

## requirements
The experiments were run with python `3.7.9`, transformers `3.1.0`, pytorch `1.7.1` using NVIDIA A100 (CUDA version `11.2`).
Download the public zeshel data [here](https://github.com/lajanugen/zeshel) [2].


## Reproducibility

### retrieval
```
python main_retriever.py --model [model saving path]  --data_dir [zeshel data directory] --B 16 --gradient_accumulation_steps 2 --logging_steps 1000 --k 64 --epochs 4 --lr 0.00001 --num_cands 64 --type_cands mixed_negative --cands_ratio 0.5   --gpus 3,4,5,7    --type_model sum_max  --num_mention_vecs 128 --num_entity_vecs 128 --store_en_hiddens --en_hidden_path [the path for saving all the entity embeddings]  --entity_bsz 4096  --mention_bsz 200

```

### save retrieved candidates
```
python save_candidates.py --model [pretrained model path] --data_dir [Zeshel data directory] --pre_model Bert --type_model sum_max --num_mention_vecs 128 --num_entity_vecs 128 --entity_bsz 1024  --mention_bsz 200 --store_en_hiddens --en_hidden_path [the path for saving all the entity embeddings]  --num_cands 64 --cands_dir [the directory for saving the candidates] --gpus 0

```


### reranking
```
python main_reranker.py --model [model saving path] --data [zeshel data directory] --B 2  --gradient_accumulation_steps 2 --num_workers 2 --warmup_proportion 0.2 --epochs 3  --gpus 5  --lr 2e-5 --cands_dir [candidates file directory]  --eval_method [micro or macro] --type_model full --type_bert [base/large]  --inputmark [--fp16]

```

## References
[1] [Understanding Hard Negatives in Noise Contrastive Estimation (Zhang and Stratos, 2021)](https://arxiv.org/pdf/2104.06245.pdf)
```
@article{zhang2021understanding,
  title={Understanding Hard Negatives in Noise Contrastive Estimation},
  author={Zhang, Wenzheng and Stratos, Karl},
  journal={arXiv preprint arXiv:2104.06245},
  year={2021}
}
```
[2] https://github.com/lajanugen/zeshel

