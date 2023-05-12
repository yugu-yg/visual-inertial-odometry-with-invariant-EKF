import os
import subprocess
from log import logger, Logger
from params import par
from eval import kitti_eval_pyimpl
import glob
import numpy as np
import prettytable


def execute(cmd):
    """ 
    执行C++程序，该程序用于评估kitti格式的绝对姿态轨迹
    ----------------------------------------------------------------------
    Args:
        cmd   : 命令行的命令，用于执行C++程序
    """
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        if "page written on" not in stdout_line and "Heiko Oberdiek" not in stdout_line:
            yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def compute_error_for_each_seq(kitti_eval_out_dir):
    """ 
    获取kitti格式轨迹的评估结果
    ----------------------------------------------------------------------
    Args:
        kitti_eval_out_dir    : kitti格式轨迹的评估结果储存目录
    Returns:
        seq_errors            : 所有参与评估的数据序列的平移error和转动error
        ave_errors[0]         : 平均平移error
        ave_errors[1]         : 平均转动error
    """
    # print the errors
    seq_error_files = os.listdir(os.path.join(kitti_eval_out_dir, "errors"))
    seq_errors = {}
    for seq_error_file in seq_error_files:
        seq = os.path.splitext(seq_error_file)[0]
        errors = np.loadtxt(os.path.join(kitti_eval_out_dir, "errors", seq_error_file))
        seq_errors[seq] = (np.average(errors[:, 2]), np.average(errors[:, 1]),)  # translation and rotation

    ave_errors = np.loadtxt(os.path.join(kitti_eval_out_dir, "stats.txt"))

    return seq_errors, (ave_errors[0], ave_errors[1],)


def print_error_table(errors, ave_errors):
    """ 
    将error加入到table中，并将其输出和记录
    ----------------------------------------------------------------------
    Args:
        errors      : 所有参与评估的数据序列的平移error和转动error
        ave_errors  : 平均error，包含平均平移error、平均转动error
    """
    table = prettytable.PrettyTable()
    table.field_names = ["Seq.", "Trans. Err", "Rot. Error"]
    table.align["Seq."] = "l"
    table.align["Trans. Err"] = "r"
    table.align["Rot. Error"] = "r"
    keys = sorted(list(errors.keys()))
    for key in keys:
        table.add_row([key, "%.6f" % errors[key][0], "%.6f" % (errors[key][1] * 180 / np.pi)])
    table.add_row(["Ave.", "%.6f" % ave_errors[0], "%.6f" % (ave_errors[1] * 180 / np.pi)])
    logger.print(table)
    logger.print("Copy to Google Sheets:",
                 ",".join([str(errors[k][0]) + "," + str(errors[k][1] * 180 / np.pi) for k in
                           sorted(list(errors.keys()))]) +
                 "," + str(ave_errors[0]) + "," + str(ave_errors[1] * 180 / np.pi))


def kitti_eval_simple(working_dir, seqs):
    """ 
    读取kitti格式的轨迹，并进行评估（简单模式）（未采用）
    --------------------------------------------------------------------------------
    Args:
        working_dir   : 工作目录，即inference步骤生成的数据储存的目录，目录名称以.traj结尾
        seqs          : 数据序列名称列表
    """
    logger.initialize(working_dir=working_dir, use_tensorboard=False)
    logger.print("================ Evaluate KITTI SIMPLE ================")
    logger.print("Working on directory:", working_dir)

    pose_est_dir = os.path.join(working_dir, "est_poses")
    pose_gt_dir = os.path.join(working_dir, "gt_poses")
    assert sorted(os.listdir(pose_est_dir)) == sorted(os.listdir(pose_gt_dir))
    pose_est_files = sorted(os.listdir(pose_est_dir))

    if seqs is None:
        seqs = [os.path.splitext(f)[0] for f in pose_est_files]

    errs = []

    for i, seq in enumerate(seqs):
        est_poses = np.load(os.path.join(pose_est_dir, "%s.npy" % seq))
        gt_poses = np.load(os.path.join(pose_gt_dir, "%s.npy" % seq))
        err = kitti_eval_pyimpl.calc_kitti_seq_errors(gt_poses, est_poses)
        errs += err
        err = np.array(err)

        if err.any():
            logger.print("Seq %s trans & rot error:" % seq)
            logger.print("%.6f   %.6f\n" % (np.average(err[:, 0]), np.average(err[:, 1]) * 180 / np.pi))
        else:
            logger.print("Error cannot be computed for seq %s\n" % seq)

    errs = np.array(errs)
    logger.print("Ave trans & rot error:")
    logger.print("%.6f   %.6f\n" % (np.average(errs[:, 0]), np.average(errs[:, 1]) * 180 / np.pi))


def kitti_eval(working_dir, train_sequences, val_sequences, min_num_frames=200):
    """ 
    读取kitti格式的轨迹，并进行评估
    轨迹指绝对姿态序列
    --------------------------------------------------------------------------------
    Args:
        working_dir      : 工作目录，即inference步骤生成的数据储存的目录，目录名称以.traj结尾
        train_sequences  : 训练集(train)的数据序列名称列表
        val_sequences    : 测试集(evaluation)的数据序列名称列表
        min_num_frames   : 参与评估的最低姿态数量(timestep+1、frame)
    """
    logger.initialize(working_dir=working_dir, use_tensorboard=False)
    logger.print("================ Evaluate KITTI ================")
    logger.print("Working on directory:", working_dir)

    # 执行评估的C++程序位置：'本项目目录/eval/kitti_eval/cpp/evaluate_odometry'
    executable = os.path.join(par.project_dir, "eval", "kitti_eval", "cpp", "evaluate_odometry")
    # kitti格式轨迹的存放目录
    kitti_dir = os.path.join(working_dir, "kitti")

    # 获取可以参与评估的数据序列名称，每个数据序列至少包含200个姿态才能参与评估
    available_seqs = []
    for f in sorted(os.listdir(kitti_dir)):
        # must have at least 200 entries for evaluation
        # 每个数据序列至少需要200个姿态参与评估
        if f.endswith(".txt") and sum(1 for line in open(os.path.join(kitti_dir, f))) > min_num_frames:
            available_seqs.append(f.replace("_est.txt", "").replace("_gt.txt", ""))

    assert (len(available_seqs) // 2 == len(set(available_seqs)))
    available_seqs = list(set(available_seqs))

    print("Poses not available for these sequences:",
          ", ".join(list(set(available_seqs) ^ set(train_sequences + val_sequences))))

    # 对训练集(train)的数据序列进行评估
    if train_sequences:
        logger.print("Evaluating training sequences...")
        # 输出参与评估的训练集(train)的数据序列名称
        logger.print("Training sequences: ", ", ".join(list(set(available_seqs) & set(train_sequences))))
        # 运行C++程序进行评估
        cmd = [executable, kitti_dir, Logger.make_dir_if_not_exist(os.path.join(kitti_dir, "train"))] + list(
                set(available_seqs) & set(train_sequences))
        # 输出并记录评估的结果
        for line in execute(cmd):
            logger.print(line.strip())

    # 对测试集(validation)的数据序列进行评估
    if val_sequences:
        logger.print("Evaluating validation sequences...")
        # 输出参与评估的测试集(validation)的数据序列名称
        logger.print("Validation sequences:", " ,".join(list(set(available_seqs) & set(val_sequences))))
        cmd = [executable, kitti_dir, Logger.make_dir_if_not_exist(os.path.join(kitti_dir, "valid"))] + list(
                set(available_seqs) & set(val_sequences))
        # 输出并记录评估结果
        execute(cmd)
        for line in execute(cmd):
            logger.print(line.strip())

    # 删除多余文件
    logger.print("Deleting useless files...")
    selected_files = list(glob.iglob(os.path.join(kitti_dir, "train", "**"), recursive=True)) + \
                     list(glob.iglob(os.path.join(kitti_dir, "valid", "**"), recursive=True))
    for filename in selected_files:
        if filename.endswith(".tex") or filename.endswith(".eps") or \
                filename.endswith(".pdf") or filename.endswith(".gp"):
            os.remove(filename)

    logger.print("Finished running KITTI evaluation!")

    # 获取所有kitti格式的绝对姿态序列的评估结果，并加入到table中，记录并输出
    # print the errors
    logger.print("Training errors are:")
    if list(set(available_seqs) & set(train_sequences)):
        train_errors, ave_train_errors = compute_error_for_each_seq(os.path.join(kitti_dir, "train"))
        print_error_table(train_errors, ave_train_errors)
    else:
        logger.print("No training errors data")

    logger.print("Validation errors are:")
    if list(set(available_seqs) & set(val_sequences)):
        val_errors, ave_val_errors = compute_error_for_each_seq(os.path.join(kitti_dir, "valid"))
        print_error_table(val_errors, ave_val_errors)
    else:
        logger.print("No validation errors data")
