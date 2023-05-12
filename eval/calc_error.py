import os
import numpy as np
from log import Logger, logger
from se3 import log_SO3, r_from_T, C_from_T


def calc_error(working_dir):
    """ 
    计算error，并将error保存到工作目录的errors文件夹中
    -----------------------------------------------------------------------------
    Args:
        working_dir    : 工作目录，即inference步骤生成的数据储存的目录，目录名称以.traj结尾
    """
    # error存放目录，工作目录下的errors文件夹中
    errors_output_dir = os.path.join(working_dir, "errors")
    # 计算error需要的数据所在目录，估计的绝对姿态和ground truth绝对姿态
    pose_est_dir = os.path.join(working_dir, "est_poses")
    pose_gt_dir = os.path.join(working_dir, "gt_poses")
    vis_meas_dir = os.path.join(working_dir, "vis_meas")
    # 获取pose_est_dir文件夹中所有npy文件名(数据序列名)
    pose_est_files = sorted(os.listdir(pose_est_dir))

    Logger.make_dir_if_not_exist(errors_output_dir)
    logger.initialize(working_dir=working_dir, use_tensorboard=False)
    logger.print("================ CALCULATE ERRORS ================")
    logger.print("Working on directory:", working_dir)
    logger.print("Found pose estimate files: \n" + "\n".join(pose_est_files))

    # 遍历所有数据序列
    for i, pose_est_file in enumerate(pose_est_files):
        # 获取数据序列名称
        sequence = os.path.splitext(pose_est_file)[0]

        # 获取该数据序列的所有估计的绝对姿态，估计的绝对姿态序列
        traj_est = np.load(os.path.join(pose_est_dir, "%s.npy" % sequence))
        # 获取该数据序列的所有ground truth的绝对姿态，gt绝对姿态序列
        traj_gt = np.load(os.path.join(pose_gt_dir, "%s.npy" % sequence))
        # 获取该数据序列的所有视觉测量值，视觉测量值序列
        vis_meas = np.load(os.path.join(vis_meas_dir, "meas", "%s.npy" % sequence))
        assert (traj_est.shape[0] == traj_gt.shape[0])

        abs_traj_error = []
        rel_traj_error = [np.zeros(6)]
        vis_meas_error = [np.zeros(6)]

        # 计算绝对姿态error
        for j in range(0, traj_est.shape[0]):
            pose_est = traj_est[j]
            pose_gt = traj_gt[j]
            # 计算绝对姿态error
            abs_pose_err = np.linalg.inv(pose_est).dot(pose_gt)
            # 保存绝对姿态error，error以旋转矩阵C和平移量r表示
            abs_traj_error.append(np.concatenate([log_SO3(C_from_T(abs_pose_err)), r_from_T(abs_pose_err)]))

        # 计算相对姿态error
        for j in range(1, traj_est.shape[0]):
            # 计算gt相对姿态(通过gt的绝对姿态)
            rel_pose_gt = np.linalg.inv(traj_gt[j - 1]).dot(traj_gt[j])
            # 计算估计的相对姿态(通过估计的绝对姿态)
            rel_pose_est = np.linalg.inv(traj_est[j - 1]).dot(traj_est[j])
            # 计算相对姿态error
            rel_pose_err = np.linalg.inv(rel_pose_est).dot(rel_pose_gt)
            # 保存相对姿态error，error以旋转矩阵C和平移量r表示
            rel_traj_error.append(np.concatenate([log_SO3(C_from_T(rel_pose_err)), r_from_T(rel_pose_err)]))
            # 保存视觉测量值error，error以旋转矩阵C(为gt相对姿态的C阵)和平移量r(为gt相对姿态的r - 前一个视觉测量值)表示
            vis_meas_error.append(np.concatenate([log_SO3(C_from_T(rel_pose_gt)), r_from_T(rel_pose_gt)]) -
                                  vis_meas[j - 1])

        # 保存该数据序列所有timestep的绝对姿态error
        np.save(logger.ensure_file_dir_exists(os.path.join(errors_output_dir, "abs", sequence + ".npy")),
                np.array(abs_traj_error))
        # 保存该数据序列所有timestep的相对姿态error
        np.save(logger.ensure_file_dir_exists(os.path.join(errors_output_dir, "rel", sequence + ".npy")),
                np.array(rel_traj_error))
        # 保存该数据序列所有timestep的视觉测量值error
        np.save(logger.ensure_file_dir_exists(os.path.join(errors_output_dir, "vis_meas", sequence + ".npy")),
                np.array(vis_meas_error))
        logger.print("Error saved for sequence %s" % sequence)

    logger.print("All Done.")
