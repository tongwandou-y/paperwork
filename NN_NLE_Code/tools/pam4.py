import numpy as np
import matplotlib.pyplot as plt

# --- 可调参数 ---

NUM_SYMBOLS = 30000  # 要生成的符号数量（越多，"云"越清晰）
NOISE_STD_DEV = 0.2  # 噪声的标准差（值越大，"云"越分散）


# --- 脚本主干 ---

def plot_pam4_constellation():
    """
    生成并绘制 PAM4 信号的星座图。
    """

    # 1. 定义理想的 PAM4 星座点
    # PAM (Pulse Amplitude Modulation) 是一种一维调制
    # 信号只在 I (In-phase) 轴上取值，Q (Quadrature) 轴理想情况下为 0
    # 四个电平通常对称设置为 -3, -1, 1, 3
    pam4_levels = np.array([-3, -1, 1, 3])

    # 2. 生成随机的发送符号
    # 从四个电平中随机选择 NUM_SYMBOLS 个符号
    tx_symbols_i = np.random.choice(pam4_levels, size=NUM_SYMBOLS)

    # 理想的 Q 分量为 0
    tx_symbols_q = np.zeros(NUM_SYMBOLS)

    # 3. 添加加性高斯白噪声 (AWGN)
    # 模拟真实信道中的噪声
    # 噪声同时影响 I 和 Q 分量
    noise_i = np.random.normal(0, NOISE_STD_DEV, size=NUM_SYMBOLS)
    noise_q = np.random.normal(0, NOISE_STD_DEV, size=NUM_SYMBOLS)

    # 4. 计算接收到的符号
    # 接收信号 = 发送信号 + 噪声
    rx_symbols_i = tx_symbols_i + noise_i
    rx_symbols_q = tx_symbols_q + noise_q

    # 5. 绘制星座图
    print(f"正在绘制 {NUM_SYMBOLS} 个 PAM4 符号...")

    plt.figure(figsize=(10, 8))

    # 绘制接收到的（带噪声的）符号
    # 使用 'alpha' 参数设置透明度，以显示点的密度
    # 使用 's' 参数设置点的大小
    plt.scatter(rx_symbols_i, rx_symbols_q, s=2, alpha=0.2,
                color='blue', label='接收符号 (Rx)')

    # 绘制理想的（无噪声的）星座点
    # 用红色的 'x' 标记
    plt.scatter(pam4_levels, [0, 0, 0, 0], color='red', marker='x',
                s=150, linewidth=3, label='理想符号 (Tx)')

    # --- 格式化图表 ---
    plt.title(f'PAM4 星座图 (噪声标准差: {NOISE_STD_DEV})', fontsize=16)
    plt.xlabel('I (同相分量)', fontsize=12)
    plt.ylabel('Q (正交分量)', fontsize=12)

    # 添加网格和中心轴线
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.7)

    # 设置图例
    plt.legend()

    # 关键：设置 'equal' 轴比例
    # 这能确保 I 轴和 Q 轴的尺度相同，
    # 使得圆形的噪声云（I 和 Q 噪声方差相同）显示为圆形，而不是椭圆形。
    plt.axis('equal')

    # 调整坐标轴范围，使其更美观
    max_val = np.max(pam4_levels) + 1
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val / 2, max_val / 2)

    plt.tight_layout()

    # 保存图像文件
    output_filename = '../pam4_constellation_diagram.png'
    plt.savefig(output_filename)
    print(f"星座图已保存为: {output_filename}")

    # 显示图像
    plt.show()


if __name__ == "__main__":

    # 确保安装了所需库
    try:
        import numpy
        import matplotlib
    except ImportError:
        print("错误：需要 numpy 和 matplotlib 库。")
        print("请使用 'pip install numpy matplotlib' 命令安装。")
        exit()

    plot_pam4_constellation()