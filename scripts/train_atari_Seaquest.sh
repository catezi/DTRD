python run_atari.py --seed 123 --context_length 30 --epochs 100 --game 'Seaquest' \
--batch_size 64 --drop_out 0.1 --learning_rate 1e-3 --redistribute_learning_rate 1e-3 \
--redistribute_step_size 1000 --redistribute_gamma 0.9 --trajectory_lamb 1e-2 \
--n_layer 2 --n_head 8 --n_embd 128 --discrete_redistribute 1 \
--save 1 --save_model_path './trained_model/Seaquest/' --save_model_name 'policy_redistribute_model.pth' \
--data_dir './game_data/Seaquest/'