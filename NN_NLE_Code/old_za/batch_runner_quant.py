import os
import sys
import types
import time
import torch
import ml_collections  # 必须安装: pip install ml_collections

# ==============================================================================
# [关键步骤 1] 模块欺骗 (Module Mocking)
# 因为你删除了 configs.py，如果 train.py 里写了 "import configs"，程序会直接报错找不到文件。
# 下面这几行代码会在内存里创建一个假的 "configs" 模块，防止 train.py 报错。
# ==============================================================================
dummy_configs = types.ModuleType("configs")
sys.modules["configs"] = dummy_configs

# ==============================================================================
# [关键步骤 2] 导入训练和测试函数
# ==============================================================================
try:
    from train import train
    from run_equalization import run_equalization
except ImportError as e:
    print("❌ 导入错误: 请确保 train.py 和 run_equalization.py 在当前目录下。")
    raise e


# ==============================================================================
# [核心功能] 动态配置生成器
# 这里的参数完全复刻了你原版 configs.py 的内容，但路径变成了动态生成的。
# ==============================================================================
def generate_config(quant_bits, target_power):
    config = ml_collections.ConfigDict()

    # --- 1. 基础物理与模型设置 (保持不变) ---
    config.fiber_length = '30km'
    config.model_type = 'DNN'
    config.quant = quant_bits  # 动态量化比特

    # 接收光功率
    config.received_optical_power = target_power

    train_prbs = 'PRBS23'
    test_prbs = 'PRBS31'

    # --- 2. 目录路径构建 (严格按照你的要求) ---
    # 根路径: E:\yinshibo\paperwork\Experiment_Data\Quant\20Gsyms_30km_{quant}bit
    root_base_dir = fr"E:\yinshibo\paperwork\Experiment_Data\Quant\20Gsyms_30km_{quant_bits}bit"

    # [A] 输入目录 (.mat 文件所在位置)
    input_dir = os.path.join(root_base_dir, "NN_Input_Data_mat")

    # [B] 输出目录 (.mat 结果)
    output_dir = os.path.join(root_base_dir, "NN_Output_Data_mat")

    # [C] 日志目录 (Loss日志, 模型, 图片)
    loss_log_dir = os.path.join(root_base_dir, "NN_Loss_Log_txt")

    # [D] 检查点目录 (Checkpoint)
    checkpoint_dir = os.path.join(root_base_dir, "NN_Train_Checkpoint_ckpt")

    # 自动创建不存在的目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(loss_log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- 3. 具体文件路径构造 ---

    # 训练输入文件: Data_For_NN_PRBS23_{quant}bit_train_{power}.mat
    config.train_data_file = os.path.join(
        input_dir,
        f'Data_For_NN_{train_prbs}_{quant_bits}bit_train_{target_power}.mat'
    )

    # 测试输入文件: Data_For_NN_PRBS31_{quant}bit_test_{power}.mat
    config.test_data_file = os.path.join(
        input_dir,
        f'Data_For_NN_{test_prbs}_{quant_bits}bit_test_{target_power}.mat'
    )

    # 测试输出文件: NN_Output_test_DNN_{quant}bit_{power}.mat
    config.test_output_file = os.path.join(
        output_dir,
        f'NN_Output_test_{config.model_type}_{quant_bits}bit_{target_power}.mat'
    )

    # 实验名称: 16QAM-30km-DNN-{quant}bit_{power}dBm
    config.experiment_name = f'16QAM-{config.fiber_length}-{config.model_type}-{quant_bits}bit_{target_power}dBm'

    # Loss 日志文件: Loss_Log_16QAM-30km-DNN-{quant}bit_{power}dBm.txt
    config.loss_log_file = os.path.join(
        loss_log_dir,
        f'Loss_Log_{config.experiment_name}.txt'
    )

    # 模型保存路径 (Best Model)
    config.model_save_path = os.path.join(
        loss_log_dir,
        f'BestModel_{config.experiment_name}.pth'
    )

    # Loss 图片保存路径
    config.loss_save_path = os.path.join(
        loss_log_dir,
        f'LossCurve_{config.experiment_name}.png'
    )

    # 注入 Checkpoint 目录 (train.py 需要使用 config.checkpoint_dir)
    config.checkpoint_dir = checkpoint_dir

    # --- 4. 训练超参 (绝对不能变) ---
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.batch_size = 64
    config.epoch = 100
    config.learn_rate = 0.001
    config.init_type = 'orthogonal'

    # --- 5. 均衡器参数 (绝对不能变) ---
    config.seq_len = 8  # 这里的 8 对应原来的 configs.py

    # 为了兼容部分代码可能用到 path 别名，做个冗余备份
    config.train_data_path = config.train_data_file
    config.test_data_path = config.test_data_file
    config.mat_save_path = config.test_output_file
    config.log_save_path = config.loss_log_file

    return config


# ==============================================================================
# [主程序] 批量运行逻辑
# ==============================================================================
def main():
    # ---------------- 配置区域 ----------------
    # 1. 设置要遍历的量化比特数
    quant_list = range(3, 13)  # [3, 4, ... 10]

    # 2. 设置要遍历的接收光功率
    power_list = range(-20, -21, -1)  # [-20, -21]
    # -----------------------------------------

    total_start_time = time.time()
    print(f"🚀 [全内建配置模式] 启动任务")
    print(f"🎯 量化比特列表: {list(quant_list)}")
    print(f"🎯 接收功率列表: {list(power_list)}\n")

    for quant in quant_list:
        print(f"\n{'=' * 80}")
        print(f"📦 [Layer 1] 进入量化精度: {quant}-bit")
        print(f"{'=' * 80}")

        for power in power_list:
            loop_start = time.time()
            print(f"\n   {'-' * 60}")
            print(f"   ▶️  [Layer 2] 处理: [{quant}-bit] @ [{power} dBm]")
            print(f"   {'-' * 60}")

            try:
                # 1. 动态生成专属配置对象
                current_config = generate_config(quant, power)

                # 2. 检查输入文件是否存在
                if not os.path.exists(current_config.train_data_file):
                    print(f"   ⚠️  [Error] 输入文件不存在!")
                    print(f"      期待路径: {current_config.train_data_file}")
                    print(f"      (请检查 MATLAB 是否已生成 {quant}bit 的 .mat 数据)")
                    print("   ⏭️  跳过...")
                    continue

                print(f"   📂 读取: {os.path.basename(current_config.train_data_file)}")
                print(f"   📂 输出: {os.path.basename(current_config.test_output_file)}")
                print(f"   📂 日志: {os.path.dirname(current_config.loss_log_file)}")
                print(f"   📂 CKPT: {current_config.checkpoint_dir}")

                # 3. 执行训练
                print("   [Step 1] 开始训练 (Train)...")
                train(current_config)

                # 4. 执行均衡测试
                print("   [Step 2] 开始均衡 (Equalization)...")
                run_equalization(current_config)

                print(f"   ✅ 成功完成: [{quant}bit | {power}dBm]")

            except Exception as e:
                print(f"\n   ❌ [CRASH] 任务崩溃: {e}")
                import traceback
                traceback.print_exc()

            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"   ⏱️  耗时: {time.time() - loop_start:.2f} 秒")

    print(f"\n{'=' * 80}")
    print(f"🏁 所有任务已完成。总耗时: {(time.time() - total_start_time) / 60:.2f} 分钟")


if __name__ == '__main__':
    main()