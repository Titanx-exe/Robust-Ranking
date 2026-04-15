import subprocess

#from run_script_E5_msmarco import commands




commands =["python3 optuna_optimize.py --found_model e5 --type_optimization all_encoder_layers_e5 --dataset aida --learning_rate 3e-6 --gpu_id 0 --noise_ratio 0 --agce yes",
            "python3 optuna_optimize.py --found_model e5 --type_optimization all_encoder_layers_e5 --dataset aida --learning_rate 3e-6 --gpu_id 0 --noise_ratio 1 --agce yes",
           "python3 optuna_optimize.py --found_model e5 --type_optimization all_encoder_layers_e5 --dataset aida --learning_rate 3e-6 --gpu_id 0 --noise_ratio 2 --agce yes",
           "python3 optuna_optimize.py --found_model e5 --type_optimization all_encoder_layers_e5 --dataset aida --learning_rate 3e-6 --gpu_id 0 --noise_ratio 3 --agce yes",
           "python3 optuna_optimize.py --found_model e5 --type_optimization all_encoder_layers_e5 --dataset aida --learning_rate 3e-6 --gpu_id 0 --noise_ratio 4 --agce yes",
           "python3 optuna_optimize.py --found_model e5 --type_optimization all_encoder_layers_e5 --dataset aida --learning_rate 3e-6 --gpu_id 0 --noise_ratio 5 --agce yes",
           "python3 optuna_optimize.py --found_model e5 --type_optimization all_encoder_layers_e5 --dataset aida --learning_rate 3e-6 --gpu_id 0 --noise_ratio 0 --agce yes",
           "python3 optuna_optimize.py --found_model e5 --type_optimization all_encoder_layers_e5 --dataset aida --learning_rate 3e-6 --gpu_id 0 --noise_ratio 1 --agce yes",
           "python3 optuna_optimize.py --found_model e5 --type_optimization all_encoder_layers_e5 --dataset aida --learning_rate 3e-6 --gpu_id 0 --noise_ratio 2 --agce yes",
           "python3 optuna_optimize.py --found_model e5 --type_optimization all_encoder_layers_e5 --dataset aida --learning_rate 3e-6 --gpu_id 0 --noise_ratio 3 --agce yes",
           "python3 optuna_optimize.py --found_model e5 --type_optimization all_encoder_layers_e5 --dataset aida --learning_rate 3e-6 --gpu_id 0 --noise_ratio 4 --agce yes",
           "python3 optuna_optimize.py --found_model e5 --type_optimization all_encoder_layers_e5 --dataset aida --learning_rate 3e-6 --gpu_id 0 --noise_ratio 5 --agce yes",
           "python3 optuna_optimize.py --found_model e5 --type_optimization all_encoder_layers_e5 --dataset aida --learning_rate 3e-6 --gpu_id 0 --noise_ratio 0 --agce yes",
           "python3 optuna_optimize.py --found_model e5 --type_optimization all_encoder_layers_e5 --dataset aida --learning_rate 3e-6 --gpu_id 0 --noise_ratio 1 --agce yes",
           "python3 optuna_optimize.py --found_model e5 --type_optimization all_encoder_layers_e5 --dataset aida --learning_rate 3e-6 --gpu_id 0 --noise_ratio 2 --agce yes",
           "python3 optuna_optimize.py --found_model e5 --type_optimization all_encoder_layers_e5 --dataset aida --learning_rate 3e-6 --gpu_id 0 --noise_ratio 3 --agce yes",
           "python3 optuna_optimize.py --found_model e5 --type_optimization all_encoder_layers_e5 --dataset aida --learning_rate 3e-6 --gpu_id 0 --noise_ratio 4 --agce yes",
           "python3 optuna_optimize.py --found_model e5 --type_optimization all_encoder_layers_e5 --dataset aida --learning_rate 3e-6 --gpu_id 0 --noise_ratio 5 --agce yes"]


'''
commands = [
    "python3 optuna_optimize.py --found_model e5 --type_optimization all_encoder_layers_e5 --learning_rate 3e-6 --gpu_id 0 --noise_ratio 0 --gce_loss yes",
    "python3 optuna_optimize.py --found_model e5 --type_optimization all_encoder_layers_e5 --learning_rate 3e-6 --gpu_id 0 --noise_ratio 1 --gce_loss yes",
    "python3 optuna_optimize.py --found_model e5 --type_optimization all_encoder_layers_e5 --learning_rate 3e-6 --gpu_id 0 --noise_ratio 2 --gce_loss yes",
    "python3 optuna_optimize.py --found_model e5 --type_optimization all_encoder_layers_e5 --learning_rate 3e-6 --gpu_id 0 --noise_ratio 3 --gce_loss yes",
    "python3 optuna_optimize.py --found_model e5 --type_optimization all_encoder_layers_e5 --learning_rate 3e-6 --gpu_id 0 --noise_ratio 4 --gce_loss yes",
    "python3 optuna_optimize.py --found_model e5 --type_optimization all_encoder_layers_e5 --learning_rate 3e-6 --gpu_id 0 --noise_ratio 5 --gce_loss yes"]
'''



for cmd in commands:
    print(f"Running: {cmd}")
    subprocess.run(cmd.split(), check=True)
