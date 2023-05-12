import numpy as np
import os
import prettytable
from evo.tools import file_interface
from evo.core import trajectory, sync, metrics
from params import par
from data_loader import SequenceData
from log import logger


# from evo.tools import log
# log.configure_logging(verbose=True, debug=True, silent=False)
#
# import numpy as np
#
# from evo.tools import plot
# import matplotlib.pyplot as plt
#
# # temporarily override some package settings
# from evo.tools.settings import SETTINGS
# SETTINGS.plot_usetex = False

def calc_euroc_seq_errors(est_traj, gt_traj):
    """ 
    根据估计的绝对姿态序列和gt的绝对姿态序列，计算绝对姿态error
    -----------------------------------------------------------
    Args:
        est_traj   : 估计的绝对姿态序列
        gt_traj    : gt的绝对姿态序列
    Returns:
        ape_metric : ？？没什么用
        ape_stat   : 绝对姿态error
    """
    # 两组姿态序列的每一个姿态进行关联，每一个姿态的绝对时间差最大不超过max_diff
    gt_traj_synced, est_traj_synced = sync.associate_trajectories(gt_traj, est_traj, max_diff=0.01)
    # 两组姿态序列对齐
    est_traj_aligned = trajectory.align_trajectory(est_traj_synced, gt_traj_synced,
                                                   correct_scale=False, correct_only_scale=False)
    # 评估两组轨迹绝对位姿误差
    pose_relation = metrics.PoseRelation.translation_part
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data((gt_traj_synced, est_traj_aligned,))
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)

    # ape_metric = metrics.RPE(pose_relation)
    # ape_metric.process_data((gt_traj_synced, est_traj_aligned,))
    # ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)

    # fig = plt.figure()
    # traj_by_label = {
    #     "estimate (not aligned)": est_traj,
    #     "estimate (aligned)": est_traj_aligned,
    #     "reference": gt_traj
    # }
    # plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
    # plt.show()

    return ape_metric, ape_stat


class EurocErrorCalc(object):
    """EuRoC数据集的error计算类"""

    def __init__(self, sequences):
        """ 
        初始化函数，读取EuRoC数据集的参与评估的数据序列的gt绝对姿态和raw时间戳
        -----------------------------------------------------------
        Args:
            sequences   : 进行评估的数据序列名称列表
        """
        # 累计error列表
        self.errors = []
        # EuRoC数据集的gt绝对姿态字典，key为数据序列的名称
        self.gt_traj = {}
        # EuRoC数据集的raw时间戳字典，key为数据序列的名称
        self.raw_timestamps = {}

        # 遍历所有数据序列
        for seq in sequences:
            # 从csv文件中读取EuRoC数据集的gt绝对姿态
            gt_traj = file_interface.read_euroc_csv_trajectory(os.path.join(par.data_dir, seq, "groundtruth.csv"))
            self.gt_traj[seq] = gt_traj
            # 记录raw时间戳
            self.raw_timestamps[seq] = np.array(SequenceData(seq).get_timestamps_raw()) / 10 ** 9

    def accumulate_error(self, seq, est):
        """ 
        计算当前数据序列的绝对姿态error，并进行累计
        ----------------------------------------------------------
        Args:
            seq       : 进行评估的单个数据序列名称
            est       : 进行评估的单个数据序列的估计绝对姿态列表
        Returns:
            ape_stat  : 当前数据序列的绝对姿态error
        """
        assert (seq in self.gt_traj)
        # 格式转换，姿态序列 + 时间戳
        est_traj = trajectory.PoseTrajectory3D(poses_se3=est, timestamps=self.raw_timestamps[seq][:len(est)])
        # 计算绝对姿态error
        ape_metric, ape_stat = calc_euroc_seq_errors(est_traj, self.gt_traj[seq])
        # 累计error
        self.errors.append(ape_stat)
        # 返回当前数据序列的绝对姿态error
        return ape_stat

    def get_average_error(self):
        """
        计算并获取绝对姿态累计error的平均值
        """
        return np.average(np.array(self.errors))

    def clear(self):
        """ 
        清除所有绝对姿态累计error
        """
        self.errors = []


def euroc_eval(working_dir, seqs):
    """ 
    读取模型基于EuRoC数据集生成的姿态，并进行评估
    轨迹指绝对姿态序列
    --------------------------------------------------------------------------------
    Args:
        working_dir      : 工作目录，即inference步骤生成的数据储存的目录，目录名称以.traj结尾
        seqs             : 进行评估的数据序列名称列表
    """
    logger.initialize(working_dir=working_dir, use_tensorboard=False)
    logger.print("================ Evaluate EUROC ================")
    logger.print("Working on directory:", working_dir)

    # 计算error需要的数据所在目录，估计的绝对姿态和ground truth绝对姿态
    pose_est_dir = os.path.join(working_dir, "est_poses")
    pose_est_files = sorted(os.listdir(pose_est_dir))

    logger.print("Evaluating seqs: %s" % ", ".join(seqs))
    # 获取该目录下评估的数据序列名称
    available_seqs = [seq.replace(".npy", "") for seq in pose_est_files]

    table = prettytable.PrettyTable()
    table.field_names = ["Seq.", "RMSE APE Err"]
    table.align["Seq."] = "l"
    table.align["RMSE APE Err"] = "r"

    assert set(seqs).issubset(available_seqs), "est file is not available, have seqs: %s" % \
                                               ", ".join(list(available_seqs))
    # 初始化EuRoC数据集的error计算类
    error_calc = EurocErrorCalc(seqs)
    stats = []

    # 遍历所有需要评估的数据序列
    for seq in seqs:
        if seq not in available_seqs:
            raise RuntimeError("File for seq %s not available" % seq)

        # 获取该数据序列的所有估计的绝对姿态，估计的绝对姿态序列
        poses_est = np.load(os.path.join(pose_est_dir, "%s.npy" % seq))
        # 计算并获取当前数据序列的绝对姿态error
        stat = error_calc.accumulate_error(seq, poses_est)
        # 计算结果加入table进行显示，保留后6位小数
        table.add_row([seq, "%.6f" % stat])
        stats.append(stat)

    # 输出绝对姿态的平均error
    table.add_row(["Ave.", "%.6f" % error_calc.get_average_error()])
    logger.print(table)

    # 完整显示所有绝对姿态error，不保留小数
    logger.print("Copy to Google Sheets:", ",".join([str(stat) for stat in stats]))
