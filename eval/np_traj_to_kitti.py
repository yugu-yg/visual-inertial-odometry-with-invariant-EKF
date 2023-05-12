import os
import numpy as np
from log import Logger, logger


def write_trj(file_handle, pose):
    """ 
    将轨迹转换为kitti格式，并写入文件
    ----------------------------------------------------------------------------
    Args:
        file_handle  : 需要写入kitti格式姿态的文件的句柄
        pose         : 单个timestep的绝对姿态
    """
    # 仅截取T阵中的C阵和r向量，即：pose = [ C , r ]
    #          a b c x
    # pose = [ d e f y ]，其中abcdefghi为C阵，xyz为r向量
    #          g h i z
    pose = np.concatenate(pose[0:3])
    # 逐行读取pose矩阵并写入文件，读取和写入顺序为：a、b、c、x、d、e、f、y、g、h、i、z
    file_handle.write(" ".join(["%f" % val for val in list(pose)]) + "\n")


def np_traj_to_kitti(working_dir):
    """ 
    将轨迹转换为kitti格式，并将转换后的轨迹保存到工作目录的kitti文件夹中
    轨迹指估计的绝对姿态序列、ground truth的绝对姿态序列
    kitti格式说明：
        将T阵中的C阵和r向量展开成单个向量，向量中第1~3、5~7、9~11项为C阵中的元素，第4、8、12项为r向量中的元素(xyz平移量)
    ----------------------------------------------------------------------------
    Args:
        working_dir  : 工作目录，即inference步骤生成的数据储存的目录，目录名称以.traj结尾
    """
    # kitti格式的轨迹存放目录，工作目录下的kitti文件夹中
    kitti_traj_output = os.path.join(working_dir, "kitti")
    # 转换前的轨迹数据所在目录，估计的绝对姿态和ground truth绝对姿态
    pose_est_dir = os.path.join(working_dir, "est_poses")
    pose_gt_dir = os.path.join(working_dir, "gt_poses")
    # 获取pose_est_dir文件夹中所有npy文件名(数据序列名)
    pose_est_files = sorted(os.listdir(pose_est_dir))

    Logger.make_dir_if_not_exist(kitti_traj_output)
    logger.initialize(working_dir=working_dir, use_tensorboard=False)
    logger.print("================ CONVERT TO KITTI ================")
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

        # 以write模式打开文件，该文件储存kitti格式的估计的绝对姿态
        kitti_est_file = open(os.path.join(kitti_traj_output, "%s_est.txt" % sequence), "w")
        # 以write模式打开文件，该文件储存kitti格式的ground truth的绝对姿态
        kitti_gt_file = open(os.path.join(kitti_traj_output, "%s_gt.txt" % sequence), "w")

        assert (traj_est.shape[0] == traj_gt.shape[0])

        # 遍历所有timestep，将估计的绝对姿态、ground truth的绝对姿态转换为kitti格式，并写入文件
        for j in range(0, traj_est.shape[0]):
            write_trj(kitti_est_file, traj_est[j, :, :])
            write_trj(kitti_gt_file, traj_gt[j, :, :])

        # 关闭文件
        kitti_est_file.close()
        kitti_gt_file.close()

    logger.print("All Done.")
