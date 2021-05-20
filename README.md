# hard-nce-el

## requirements
transformers 3.1.0, pytorch 1.7.1

## scripts

### retrieval
```

python main_retriever.py --model /dresden/users/wz283/wz283/model_summax_mixed/model.pt  --data_dir ./zeshel --B 16 --gradient_accumulation_steps 2 --logging_steps 1000 --k 64 --epochs 4 --lr 0.00001 --num_cands 64 --type_cands hard_and_random_negative --cands_ratio 0.5   --gpus 3,4,5,7    --type_model sum_max  --num_mention_vecs 128 --num_entity_vecs 128 -store_en_hiddens --en_hidden_path /dresden/users/wz283/wz283/en_hiddens_summax/  --entity_bsz 4096  --mention_bsz 200


```

