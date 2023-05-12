import os
import argparse


arg_parser = argparse.ArgumentParser(description='Train E2E VIO')
arg_parser.add_argument('--description', type=str, help="Description of this training run")
arg_parser.add_argument('--gpu_id', type=int, nargs="+", help="select the GPU to perform training on")
arg_parser.add_argument('--run_eval_only', default=False, action='store_true',
                        help="Only run evaluation in current working directory")
arg_parser.add_argument('--eval_results_dir', type=str, help="path of model want to eval")
arg_parser.add_argument('--resume_model_from', type=str, help="path of model state to resume from")
arg_parser.add_argument('--resume_optimizer_from', type=str, help="path of optimizer state to resume from")
arg_parsed = arg_parser.parse_args()
gpu_ids = arg_parsed.gpu_id

if gpu_ids:
    # 设置可用的GPU设备
    os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(i) for i in gpu_ids])
    # 输出可用的GPU设备
    print("CUDA_VISIVLE_DEVICES: %s" % os.environ["CUDA_VISIBLE_DEVICES"])