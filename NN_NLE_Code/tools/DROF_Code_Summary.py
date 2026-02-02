import os

# ==================== 用户配置区域 ====================

# 1. 指定你的项目文件夹路径 (输入路径)
#    请修改为你 DROF 项目的实际路径
PROJECT_PATH = r"D:\paperwork\Experiment_Code\DRoF_PCM_PAM4_Code"

# 2. 指定输出位置
#    填文件夹路径会自动生成 "DROF代码汇总.md"，也可以指定具体文件名
OUTPUT_PATH = r"D:\paperwork"

# 3. 是否扫描子文件夹？
#    False = 只扫描根目录 (你提到 .m 文件都在根目录，建议保持 False)
RECURSIVE = False

# 4. (如果上面是True) 需要忽略的文件夹名称
IGNORE_DIRS = ['libs', 'data', 'results', 'bin', 'obj', '.git']


# =====================================================

def read_file_content(file_path):
    """
    读取文件内容，兼容 UTF-8 和 GB18030 (Windows常见编码)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='gb18030') as f:
                return f.read()
        except:
            return f"% [Error] 无法识别文件编码: {os.path.basename(file_path)}"


def generate_markdown(project_dir, output_target):
    # --- 智能路径处理 ---
    # 如果用户给的是文件夹，自动追加文件名
    if os.path.isdir(output_target):
        output_file = os.path.join(output_target, "DRoF_PCM_PAM4的代码汇总.md")
    else:
        output_file = output_target

    if not os.path.exists(project_dir):
        print(f"❌ 错误：找不到项目路径 -> {project_dir}")
        return

    m_files = []  # 专门存储 .m 文件

    print(f"正在扫描目录: {project_dir}")
    if not RECURSIVE:
        print("模式: 仅根目录 (忽略子文件夹)")
    else:
        print(f"模式: 递归扫描 (已忽略: {IGNORE_DIRS})")

    # --- 核心扫描逻辑 ---
    for root, dirs, files in os.walk(project_dir):

        # 1. 递归控制
        if not RECURSIVE:
            if os.path.abspath(root) != os.path.abspath(project_dir):
                continue
            dirs[:] = []
        else:
            # 过滤忽略目录
            for i in range(len(dirs) - 1, -1, -1):
                if dirs[i].lower() in IGNORE_DIRS:
                    del dirs[i]

        for file in files:
            file_path = os.path.join(root, file)
            _, ext = os.path.splitext(file)
            ext = ext.lower()

            # 排除输出文件本身
            if os.path.abspath(file_path) == os.path.abspath(output_file):
                continue

            # === 修改点：只识别 .m 文件 ===
            if ext == '.m':
                m_files.append(file_path)

    # 排序
    m_files.sort(key=lambda x: os.path.basename(x))

    print(f"统计: .m 文件({len(m_files)})")

    # --- 写入 Markdown ---
    output_dir_path = os.path.dirname(output_file)
    if output_dir_path and not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    try:
        with open(output_file, 'w', encoding='utf-8') as md:

            # 写入标题
            if m_files:
                md.write("#### MATLAB 源文件\n\n")

                for path in m_files:
                    filename = os.path.basename(path)
                    content = read_file_content(path)

                    # 使用二级标题显示文件名
                    md.write(f"##### {filename}\n\n")
                    # 使用 matlab 语法高亮
                    md.write("```matlab\n")
                    md.write(content)
                    md.write("\n```\n\n")
            else:
                md.write("> 未找到任何 .m 文件\n")

        print(f"\n✅ 成功！文件已生成: {output_file}")

    except PermissionError:
        print(f"\n❌ 权限错误: 无法写入文件 {output_file}。")
    except Exception as e:
        print(f"\n❌ 错误: {e}")


if __name__ == "__main__":
    generate_markdown(PROJECT_PATH, OUTPUT_PATH)