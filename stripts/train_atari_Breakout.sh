python run_atari.py --seed 123 --context_length 30 --epochs 100 --game 'Breakout' \
--batch_size 64 --drop_out 0.1 --learning_rate 1e-3 --redistribute_learning_rate 1e-3 \
--redistribute_step_size 1000 --redistribute_gamma 0.9 --trajectory_lamb 1e-2 \
--n_layer 2 --n_head 8 --n_embd 32 --discrete_redistribute 1 \
--save 1 --save_model_path './trained_model/Breakout/' --save_model_name 'policy_redistribute_model.pth' \
--data_dir './game_data/Breakout/'