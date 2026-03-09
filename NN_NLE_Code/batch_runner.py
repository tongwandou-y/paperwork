# DNNV3-0/batch_runner.py

import os
import torch
import time
import configs as cfg_module
from train import train
from run_equalization import run_equalization


def main():
    # ================= 配置区域 =================
    # 设置你要遍历的光功率范围
    # range(-15, -28, -1) 表示从 -15 开始，每次 -1，直到 -27 (不包含-28)
    # 即: -15, -16, ..., -27
    power_list = range(-15, -28, -1)

    # 如果只想跑几个特定的，可以用列表: power_list = [-15, -20, -25]
    # 消融实验配置（排除Volterra）
    # 每个配置会在现有根目录下自动保存到独立子目录：
    #   NN_Output_Data_mat/<ablation_tag>/
    #   NN_Loss_Log_txt/<ablation_tag>/
    ablation_profiles = [
        {
            'name': 'pam_only',
            'use_loss_pam': True,
            'use_loss_pcm': False
        },
        {
            'name': 'pam_pcm',
            'use_loss_pam': True,
            'use_loss_pcm': True
        }
    ]
    # ===========================================

    total_start_time = time.time()

    print(f"🚀 开始批量任务，将处理以下功率点: {list(power_list)}")
    print(f"🧪 消融组: {[p['name'] for p in ablation_profiles]}\n")

    for profile in ablation_profiles:
        print(f"\n{'#' * 70}")
        print(f"🧪 当前消融组: {profile['name']}")
        print(f"{'#' * 70}")

        for power in power_list:
            loop_start_time = time.time()
            print(f"{'=' * 60}")
            print(f" ▶️  正在处理接收光功率: {power} dBm")
            print(f"{'=' * 60}")

            try:
                # 1. 获取当前功率 + 当前消融组的专属配置
                current_config = cfg_module.get_config(target_power=power, ablation_profile=profile)

                print(f"配置已生成: {current_config.experiment_name}")
                print(f"输出目录: {os.path.dirname(current_config.test_output_file)}")
                print(f"日志目录: {os.path.dirname(current_config.loss_log_file)}")

                # 2. 执行训练 (Train)
                print("\n[Step 1/2] 开始训练...")
                train(current_config)

                # 3. 执行均衡/测试 (Equalization)
                print("\n[Step 2/2] 开始均衡测试...")
                run_equalization(current_config)

                print(f"\n✅ 功率 {power} dBm 处理完成!")

            except Exception as e:
                # 捕获异常，防止某一个功率点报错导致整个脚本中断
                print(f"\n❌ 错误: 处理功率 {power} dBm 时发生异常!")
                print(f"错误详情: {e}")
                print("跳过当前任务，继续下一个...\n")

            finally:
                # 4. 清理显存
                # 非常重要！防止在循环中显存越积越多导致 OOM (Out Of Memory)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                elapsed = time.time() - loop_start_time
                print(f"当前轮次耗时: {elapsed:.2f} 秒\n")

    total_time = time.time() - total_start_time
    print(f"{'=' * 60}")
    print(f"🎉 所有任务已完成！总耗时: {total_time / 60:.2f} 分钟")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()