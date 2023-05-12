import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import se3
from log import logger, Logger
import os

# 自己模型
PROPOSED_DIR = '/mnt/tecmint/data/hjl/workspace/deep_iekf_vio2/results/train_20210418-10-20-27/result_eval'
# 需要进行比对的模型
COMPARED_DIR = '/mnt/tecmint/data/hjl/workspace/deep_ekf_vio/results/train_20210203-11-50-46_kitti_best/saved_model.eval2.traj'

IMU_ONLY_DIR='/mnt/tecmint/data/hjl/workspace/deep_iekf_vio2/results/train_20210418-10-20-27/imu_only'

VISION_ONLY_DIR='/mnt/tecmint/data/hjl/workspace/deep_iekf_vio2/results/train_20210418-10-20-27/vision_only_checkpoint'



if "DISPLAY" not in os.environ:
    # 如果无法显示，则plt在后台运行
    # 此时plt.show()函数无法使用
    plt.switch_backend("Agg")

def plot_single_trajectory(proposed_dir):
    # 图像存放目录，工作目录下的figures文件夹中
    output_dir = os.path.join(proposed_dir, "compare_figures")
    # 绘制图像需要的数据所在目录，估计的绝对姿态和ground truth绝对姿态
    pose_est_dir = os.path.join(proposed_dir, "est_poses")
    pose_gt_dir = os.path.join(proposed_dir, "gt_poses")
    pose_est_files = sorted(os.listdir(pose_est_dir))

    Logger.make_dir_if_not_exist(output_dir)

    for i, pose_est_file in enumerate(pose_est_files):
        # print('Plot sequence %s' % sequence)
        # 获取数据序列名称
        sequence = os.path.splitext(pose_est_file)[0]

        # 获取该数据序列的所有估计的绝对姿态，估计的绝对姿态序列
        trajectory_proposed = np.load(os.path.join(pose_est_dir, "%s.npy" % sequence))
        # 获取该数据序列的所有ground truth的绝对姿态，gt绝对姿态序列
        trajectory_gt = np.load(os.path.join(pose_gt_dir, "%s.npy" % sequence))

        # 遍历估计的绝对姿态序列，获取每个timestep的xyz平移量
        trans_xyz_proposed = np.array([se3.r_from_T(T) for T in trajectory_proposed])
        # 遍历估计的绝对姿态序列，获取每个timestep的xyz旋转量
        rot_xyz_proposed = np.array([se3.log_SO3(se3.C_from_T(T)) for T in trajectory_proposed])

        # 遍历gt绝对姿态序列，获取每个timestep的xyz平移量
        trans_xyz_gt = np.array([se3.r_from_T(T) for T in trajectory_gt])
        # 遍历gt绝对姿态序列，获取每个timestep的xyz旋转量
        rot_xyz_gt = np.array([se3.log_SO3(se3.C_from_T(T)) for T in trajectory_gt])

        # trans_xyz_compared = np.array([se3.r_from_T(T) for T in trajectory_compared])
        # rot_xyz_compared = np.array([se3.log_SO3(se3.C_from_T(T)) for T in trajectory_compared])

        # 估计的xyz平移量和旋转量
        trans_x_proposed = trans_xyz_proposed[:, 0]
        trans_y_proposed = trans_xyz_proposed[:, 1]
        trans_z_proposed = trans_xyz_proposed[:, 2]
        rot_x_proposed = rot_xyz_proposed[:, 0]
        rot_y_proposed = rot_xyz_proposed[:, 1]
        rot_z_proposed = rot_xyz_proposed[:, 2]

        # gt的xyz平移量和旋转量
        trans_x_gt = trans_xyz_gt[:, 0]
        trans_y_gt = trans_xyz_gt[:, 1]
        trans_z_gt = trans_xyz_gt[:, 2]
        rot_x_gt = rot_xyz_gt[:, 0]
        rot_y_gt = rot_xyz_gt[:, 1]
        rot_z_gt = rot_xyz_gt[:, 2]

        # trans_x_compared = trans_xyz_compared[:, 0]
        # trans_y_compared = trans_xyz_compared[:, 1]
        # trans_z_compared = trans_xyz_compared[:, 2]
        # rot_x_compared = rot_xyz_compared[:, 0]
        # rot_y_compared = rot_xyz_compared[:, 1]
        # rot_z_compared = rot_xyz_compared[:, 2]

        # 绘制XY平面轨迹图，仅使用平移量进行绘制
        plt.clf()
        plt.plot(trans_x_proposed, trans_y_proposed, linewidth=1.0, color="r", label="Estimate")
        plt.plot(trans_x_gt, trans_y_gt, linewidth=1.0, color="b", label="Ground Truth")
        # plt.plot(trans_x_compared, trans_y_compared, linewidth=1.0, color="g", label="Deep-ekf-vio")
        plt.axis("equal")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("Seq. %s Trajectory XY Plane" % sequence)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "seq_%s_00_xy_traj.png" % sequence))

        # 绘制XZ平面轨迹图，仅使用平移量进行绘制
        plt.clf()
        plt.plot(trans_x_proposed, trans_z_proposed, linewidth=1.0, color="r", label="Estimate")
        plt.plot(trans_x_gt, trans_z_gt, linewidth=1.0, color="b", label="Ground Truth")
        # plt.plot(trans_x_compared, trans_z_compared, linewidth=1.0, color="g", label="Deep-ekf-vio")
        plt.axis("equal")
        plt.xlabel("x [m]")
        plt.ylabel("z [m]")
        plt.title("Seq. %s Trajectory XZ Plane" % sequence)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "seq_%s_01_xz_traj.png" % sequence))

        # 绘制YZ平面轨迹图，仅使用平移量进行绘制
        plt.clf()
        plt.plot(trans_y_proposed, trans_z_proposed, linewidth=1.0, color="r", label="Estimate")
        plt.plot(trans_y_gt, trans_z_gt, linewidth=1.0, color="b", label="Ground Truth")
        # plt.plot(trans_y_compared, trans_z_compared, linewidth=1.0, color="g", label="Deep-ekf-vio")
        plt.axis("equal")
        plt.xlabel("y [m]")
        plt.ylabel("z [m]")
        plt.title("Seq. %s Trajectory YZ Plane" % sequence)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "seq_%s_02_yz_traj.png" % sequence))

        # 单独绘制估计和gt的平移量和旋转量在XYZ轴的差异图
        labels = ["Trans X", "Trans Y", "Trans Z", "Rot X", "Rot Y", "Rot Z"]
        data_proposed = [trans_x_proposed, trans_y_proposed, trans_z_proposed, rot_x_proposed, rot_y_proposed, rot_z_proposed]
        data_gt = [trans_x_gt, trans_y_gt, trans_z_gt, rot_x_gt, rot_y_gt, rot_z_gt]
        # data_compared = [trans_x_compared, trans_y_compared, trans_z_compared, rot_x_compared, rot_y_compared, rot_z_compared]
        for j in range(0, len(labels)):
            plt.clf()
            plt.plot(data_proposed[j], linewidth=1.0, color="r", label="Estimate")
            plt.plot(data_gt[j], linewidth=1.0, color="b", label="Ground Truth")
            # plt.plot(data_compared[j], linewidth=1.0, color="g", label="Deep-ekf-vio")
            plt.xlabel("frame # []")
            # 加上Y轴单位
            if(labels[j].split(' ')[0]=='Trans'):
                plt.ylabel(labels[j].lower()+' [m]')
            elif(labels[j].split(' ')[0]=='Rot'):
                plt.ylabel(labels[j].lower()+' [rad]')
            plt.title("Seq. %s %s" % (sequence, labels[j]))
            plt.legend()
            plt.savefig(os.path.join(output_dir, "seq_%s_%02d_%s_plt.png" %
                                     (sequence, j + 3, "_".join(labels[j].lower().split()))))

        # logger.print("Plot saved for sequence %s" % sequence)
        print("Plot saved for sequence %s" % sequence)


def compare_plot_trajectory(imu_only_dir, vision_only_dir, proposed_dir, compared_dir):
    '''
    proposed_dir: 自己模型
    compared_dir: 需要进行比对的模型
    '''
    # 图像存放目录，工作目录下的figures文件夹中
    output_dir = os.path.join(proposed_dir, "compare_figures")

    # 绘制图像需要的数据所在目录，估计的绝对姿态和ground truth绝对姿态
    pose_est_dir = os.path.join(proposed_dir, "est_poses")
    pose_gt_dir = os.path.join(proposed_dir, "gt_poses")
    pose_compare_dir = os.path.join(compared_dir, "est_poses")
    pose_imu_only_dir = os.path.join(imu_only_dir, "est_poses")
    pose_vision_only_dir = os.path.join(vision_only_dir, "est_poses")

    assert sorted(os.listdir(pose_est_dir)) == sorted(os.listdir(pose_gt_dir))
    assert sorted(os.listdir(pose_est_dir)) == sorted(os.listdir(pose_compare_dir))

    # 获取pose_est_dir文件夹中所有npy文件名(数据序列名)
    pose_est_files = sorted(os.listdir(pose_est_dir))

    Logger.make_dir_if_not_exist(output_dir)
    # logger.initialize(working_dir=working_dir, use_tensorboard=False)
    # logger.print("================ PLOT TRAJECTORY ================")
    # logger.print("Working on directory:", working_dir)
    # logger.print("Found pose estimate files: \n" + "\n".join(pose_est_files))


    for i, pose_est_file in enumerate(pose_est_files):
        # print('Plot sequence %s' % sequence)
        # 获取数据序列名称
        sequence = os.path.splitext(pose_est_file)[0]

        # 获取该数据序列的所有估计的绝对姿态，估计的绝对姿态序列
        trajectory_proposed = np.load(os.path.join(pose_est_dir, "%s.npy" % sequence))
        trans_xyz_proposed = np.array([se3.r_from_T(T) for T in trajectory_proposed])
        rot_xyz_proposed = np.array([se3.log_SO3(se3.C_from_T(T)) for T in trajectory_proposed])
        trans_x_proposed = trans_xyz_proposed[:, 0]
        trans_y_proposed = trans_xyz_proposed[:, 1]
        trans_z_proposed = trans_xyz_proposed[:, 2]
        rot_x_proposed = rot_xyz_proposed[:, 0]
        rot_y_proposed = rot_xyz_proposed[:, 1]
        rot_z_proposed = rot_xyz_proposed[:, 2]


        trajectory_gt = np.load(os.path.join(pose_gt_dir, "%s.npy" % sequence))
        trans_xyz_gt = np.array([se3.r_from_T(T) for T in trajectory_gt])
        rot_xyz_gt = np.array([se3.log_SO3(se3.C_from_T(T)) for T in trajectory_gt])
        trans_x_gt = trans_xyz_gt[:, 0]
        trans_y_gt = trans_xyz_gt[:, 1]
        trans_z_gt = trans_xyz_gt[:, 2]
        rot_x_gt = rot_xyz_gt[:, 0]
        rot_y_gt = rot_xyz_gt[:, 1]
        rot_z_gt = rot_xyz_gt[:, 2]


        trajectory_compared = np.load(os.path.join(pose_compare_dir, "%s.npy" % sequence))
        trans_xyz_compared = np.array([se3.r_from_T(T) for T in trajectory_compared])
        rot_xyz_compared = np.array([se3.log_SO3(se3.C_from_T(T)) for T in trajectory_compared])
        trans_x_compared = trans_xyz_compared[:, 0]
        trans_y_compared = trans_xyz_compared[:, 1]
        trans_z_compared = trans_xyz_compared[:, 2]
        rot_x_compared = rot_xyz_compared[:, 0]
        rot_y_compared = rot_xyz_compared[:, 1]
        rot_z_compared = rot_xyz_compared[:, 2]


        trajectory_imu_only = np.load(os.path.join(pose_imu_only_dir, "%s.npy" % sequence))
        trans_xyz_imu_only = np.array([se3.r_from_T(T) for T in trajectory_imu_only])
        rot_xyz_imu_only = np.array([se3.log_SO3(se3.C_from_T(T)) for T in trajectory_imu_only])
        trans_x_imu_only = trans_xyz_imu_only[:, 0]
        trans_y_imu_only = trans_xyz_imu_only[:, 1]
        trans_z_imu_only = trans_xyz_imu_only[:, 2]
        rot_x_imu_only = rot_xyz_imu_only[:, 0]
        rot_y_imu_only = rot_xyz_imu_only[:, 1]
        rot_z_imu_only = rot_xyz_imu_only[:, 2]


        trajectory_vision_only = np.load(os.path.join(pose_vision_only_dir, "%s.npy" % sequence))
        trans_xyz_vision_only = np.array([se3.r_from_T(T) for T in trajectory_vision_only])
        rot_xyz_vision_only = np.array([se3.log_SO3(se3.C_from_T(T)) for T in trajectory_vision_only])
        trans_x_vision_only = trans_xyz_vision_only[:, 0]
        trans_y_vision_only = trans_xyz_vision_only[:, 1]
        trans_z_vision_only = trans_xyz_vision_only[:, 2]
        rot_x_vision_only = rot_xyz_vision_only[:, 0]
        rot_y_vision_only = rot_xyz_vision_only[:, 1]
        rot_z_vision_only = rot_xyz_vision_only[:, 2]

        

        

        # 绘制XY平面轨迹图，仅使用平移量进行绘制
        plt.clf()
        plt.plot(trans_x_imu_only, trans_y_imu_only, linewidth=1.0, color="cyan", label="IMU")
        plt.plot(trans_x_vision_only, trans_y_vision_only, linewidth=1.0, color="gold", label="Vision")
        plt.plot(trans_x_compared, trans_y_compared, linewidth=1.0, color="g", label="Deep-EKF-VIO")
        plt.plot(trans_x_gt, trans_y_gt, linewidth=1.0, color="b", label="Ground Truth")
        plt.plot(trans_x_proposed, trans_y_proposed, linewidth=1.0, color="r", label="Proposed")
        plt.axis("equal")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("Seq. %s Trajectory XY Plane" % sequence)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "seq_%s_00_xy_traj.png" % sequence))
        plt.savefig(os.path.join(output_dir, "seq_%s_00_xy_traj.eps" % sequence))


        # 绘制XZ平面轨迹图，仅使用平移量进行绘制
        plt.clf()
        plt.plot(trans_x_imu_only, trans_z_imu_only, linewidth=1.0, color="cyan", label="IMU")
        plt.plot(trans_x_vision_only, trans_z_vision_only, linewidth=1.0, color="gold", label="Vision")
        plt.plot(trans_x_compared, trans_z_compared, linewidth=1.0, color="g", label="Deep-EKF-VIO")
        plt.plot(trans_x_gt, trans_z_gt, linewidth=1.0, color="b", label="Ground Truth")
        plt.plot(trans_x_proposed, trans_z_proposed, linewidth=1.0, color="r", label="Proposed")
        plt.axis("equal")
        plt.xlabel("x [m]")
        plt.ylabel("z [m]")
        plt.title("Seq. %s Trajectory XZ Plane" % sequence)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "seq_%s_01_xz_traj.png" % sequence))
        plt.savefig(os.path.join(output_dir, "seq_%s_01_xz_traj.eps" % sequence))



        # 绘制YZ平面轨迹图，仅使用平移量进行绘制
        plt.clf()
        plt.plot(trans_y_imu_only, trans_z_imu_only, linewidth=1.0, color="cyan", label="IMU")
        plt.plot(trans_y_vision_only, trans_z_vision_only, linewidth=1.0, color="gold", label="Vision")
        plt.plot(trans_y_compared, trans_z_compared, linewidth=1.0, color="g", label="Deep-EKF-VIO") 
        plt.plot(trans_y_gt, trans_z_gt, linewidth=1.0, color="b", label="Ground Truth")
        plt.plot(trans_y_proposed, trans_z_proposed, linewidth=1.0, color="r", label="Proposed")
        plt.axis("equal")
        plt.xlabel("y [m]")
        plt.ylabel("z [m]")
        plt.title("Seq. %s Trajectory YZ Plane" % sequence)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "seq_%s_02_yz_traj.png" % sequence))
        plt.savefig(os.path.join(output_dir, "seq_%s_02_yz_traj.eps" % sequence))

        # 单独绘制估计和gt的平移量和旋转量在XYZ轴的差异图
        labels = ["Trans X", "Trans Y", "Trans Z", "Rot X", "Rot Y", "Rot Z"]
        data_proposed = [trans_x_proposed, trans_y_proposed, trans_z_proposed, rot_x_proposed, rot_y_proposed, rot_z_proposed]
        data_gt = [trans_x_gt, trans_y_gt, trans_z_gt, rot_x_gt, rot_y_gt, rot_z_gt]
        data_compared = [trans_x_compared, trans_y_compared, trans_z_compared, rot_x_compared, rot_y_compared, rot_z_compared]
        data_imu_only = [trans_x_imu_only, trans_y_imu_only, trans_z_imu_only, rot_x_imu_only, rot_y_imu_only, rot_z_imu_only]
        data_vision_only = [trans_x_vision_only, trans_y_vision_only, trans_z_vision_only, rot_x_vision_only, rot_y_vision_only, rot_z_vision_only]
        for j in range(0, len(labels)):
            plt.clf()
            plt.plot(data_imu_only[j], linewidth=1.0, color="cyan", label="IMU")
            plt.plot(data_vision_only[j], linewidth=1.0, color="gold", label="Vision")
            plt.plot(data_compared[j], linewidth=1.0, color="g", label="Deep-EKF-VIO")
            plt.plot(data_gt[j], linewidth=1.0, color="b", label="Ground Truth")
            plt.plot(data_proposed[j], linewidth=1.0, color="r", label="Proposed")
            plt.xlabel("frame # []")
            # 加上Y轴单位
            if(labels[j].split(' ')[0]=='Trans'):
                plt.ylabel(labels[j].lower()+' [m]')
            elif(labels[j].split(' ')[0]=='Rot'):
                plt.ylabel(labels[j].lower()+' [rad]')
            plt.title("Seq. %s %s" % (sequence, labels[j]))
            plt.legend()
            plt.savefig(os.path.join(output_dir, "seq_%s_%02d_%s_plt.png" %
                                     (sequence, j + 3, "_".join(labels[j].lower().split()))))
            plt.savefig(os.path.join(output_dir, "seq_%s_%02d_%s_plt.eps" %
                                     (sequence, j + 3, "_".join(labels[j].lower().split()))))

        # logger.print("Plot saved for sequence %s" % sequence)
        print("Plot saved for sequence %s" % sequence)


if __name__ == "__main__":

    if "DISPLAY" not in os.environ:
        # 如果无法显示，则plt在后台运行
        # 此时plt.show()函数无法使用
        plt.switch_backend("Agg")
    
    # params = {
    #     'mathtext.fontset': 'stix',
    #     'mathtext.rm': 'serif',
    #     'font.family': 'serif',
    #     'font.serif': "Times New Roman",
    #     'figure.dpi': 100.0,
    #     'xtick.direction': 'in',
    #     'xtick.top': False,
    #     'ytick.right': False,
    #     'ytick.direction': 'in'
    # }
    # matplotlib.rcParams.update(params)

    compare_plot_trajectory(IMU_ONLY_DIR,VISION_ONLY_DIR,PROPOSED_DIR, COMPARED_DIR)


    # plot_single_trajectory(PROPOSED_DIR)


