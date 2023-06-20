python eval_atari.py --seed 123 --context_length 30 --game_episodes_num 10 --game 'Breakout' \
--drop_out 0.1 --n_layer 2 --n_head 8 --n_embd 32 --discrete_redistribute 1 --use_sample_policy 0 --min_length 20000 \
--model_path "/home/LAB/qiuyue/reward_redistribution/atari/trained_model/Breakout/epoch8_policy_redistribute_model.pth"