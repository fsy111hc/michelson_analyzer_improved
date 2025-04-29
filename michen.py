import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
import io
import base64
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def calculate_wavelength(fringes, positions):
    """
    使用线性拟合计算波长
    
    参数:
    fringes -- 干涉环圈数列表
    positions -- 位移距离列表 (mm)
    
    返回:
    wavelength -- 计算得到的波长(nm)
    wavelength_uncertainty -- 波长标准不确定度(nm)
    r_squared -- 拟合优度R²
    """
    # 转换为numpy数组以便进行计算
    fringes = np.array(fringes)
    positions = np.array(positions)
    
    # 线性回归拟合
    slope, intercept, r_value, p_value, std_err = stats.linregress(fringes, positions)
    
    # 波长 = 斜率 * 2 * 1000000 (转换为纳米)
    wavelength = slope * 2 * 1000000
    wavelength_uncertainty = std_err * 2 * 1000000
    r_squared = r_value**2
    
    return wavelength, wavelength_uncertainty, slope, std_err, r_squared

def calculate_uncertainty(slope, std_err, fringes_range):
    """
    计算波长的测量不确定度(考虑圈数读取不确定度)
    
    参数:
    slope -- 拟合得到的斜率
    std_err -- 斜率的标准不确定度
    fringes_range -- 总的干涉环圈数范围
    
    返回:
    total_uncertainty -- 总的相对不确定度(%)
    """
    # 圈数读取不确定度为0.5圈
    fringe_uncertainty = 0.5 / fringes_range
    
    # 斜率测量的相对不确定度
    slope_uncertainty = std_err / slope
    
    # 总的相对不确定度(平方和的平方根)
    total_uncertainty = np.sqrt(slope_uncertainty**2 + fringe_uncertainty**2) * 100
    
    return total_uncertainty

def correct_backlash_error(positions, fringes):
    """
    校正螺纹空程差导致的系统不确定度
    通过分析前几组数据的间隔来识别并校正
    
    参数:
    positions -- 位移距离列表 (mm)
    fringes -- 干涉环圈数列表
    
    返回:
    corrected_positions -- 校正后的位移距离列表
    """
    if len(positions) < 4:
        return positions  # 数据点太少，无法校正
    
    # 计算相邻数据间的差值
    diffs = np.diff(positions)
    
    # 检查前3个差值是否比后面的差值明显偏大
    avg_later_diffs = np.mean(diffs[3:]) if len(diffs) > 3 else np.mean(diffs)
    correction_needed = False
    
    for i in range(min(3, len(diffs))):
        if diffs[i] > avg_later_diffs * 1.05:  # 如果差值比平均值大5%以上
            correction_needed = True
            break
    
    if correction_needed:
        # 使用后面稳定部分的平均间隔来校正前面的数据
        corrected_positions = positions.copy()
        for i in range(1, min(4, len(positions))):
            ideal_position = positions[0] + i * avg_later_diffs
            corrected_positions[i] = ideal_position
        return corrected_positions
    else:
        return positions

def correct_path_error(fringes, positions, deviation_cm=0, path_length_cm=41):
    """
    校正光路偏移导致的系统不确定度
    
    参数:
    fringes -- 干涉环圈数列表
    positions -- 位移距离列表 (mm)
    deviation_cm -- 条纹中心的偏移量(cm)，默认为0
    path_length_cm -- S1到毛玻璃屏的距离(cm)，默认为41cm
    
    返回:
    corrected_wavelength -- 校正后的波长
    corrected_wavelength_uncertainty -- 校正后的波长标准不确定度
    """
    if deviation_cm == 0:
        return calculate_wavelength(fringes, positions)
    
    # 计算偏移角θ
    theta = np.arctan(deviation_cm / path_length_cm)
    cos_theta = np.cos(theta)
    
    # 校正位移数据
    corrected_positions = positions / cos_theta
    
    return calculate_wavelength(fringes, corrected_positions)

def plot_data_and_fit(fringes, positions, slope, intercept, wavelength, wavelength_uncertainty, r_squared, total_uncertainty):
    """
    绘制数据点和拟合直线
    
    返回:
    fig -- matplotlib图形对象
    """
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # 绘制原始数据点
    ax.scatter(fringes, positions, color='blue', label='实验数据')
    
    # 绘制拟合直线
    x_fit = np.linspace(min(fringes), max(fringes), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, color='red', label='线性拟合')
    
    # 添加标签和标题
    ax.set_xlabel('干涉环圈数')
    ax.set_ylabel('镜面位移 (mm)')
    ax.set_title('迈克尔逊干涉实验数据分析')
    
    # 在图中添加拟合结果
    result_text = f"波长 = {wavelength:.2f} ± {wavelength_uncertainty:.2f} nm\n"
    result_text += f"相对不确定度 = {total_uncertainty:.2f}%\n"
    result_text += f"斜率 = {slope:.8f} mm/圈\n"
    result_text += f"R² = {r_squared:.6f}"
    ax.annotate(result_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                va='top')
    
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    
    return fig

def fig_to_base64(fig):
    """将matplotlib图形转换为base64编码"""
    buf = io.BytesIO()
    FigureCanvas(fig).print_png(buf)
    img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_data

def analyze_data(fringes, positions, deviation_cm=0, path_length_cm=41, correct_backlash=True):
    """
    分析数据并返回结果和图表
    
    参数:
    fringes -- 干涉环圈数列表
    positions -- 位移距离列表 (mm)
    deviation_cm -- 条纹中心的偏移量(cm)
    path_length_cm -- S1到毛玻璃屏的距离(cm)
    correct_backlash -- 是否校正螺纹空程差
    
    返回:
    results -- 包含分析结果的字典
    fig_base64 -- base64编码的图表
    """
    # 数据准备
    fringes = np.array(fringes)
    positions = np.array(positions)
    
    # 校正螺纹空程差(如果需要)
    if correct_backlash:
        positions = correct_backlash_error(positions, fringes)
    
    # 计算波长(考虑光路偏移)
    if deviation_cm != 0:
        wavelength, wavelength_uncertainty, slope, std_err, r_squared = correct_path_error(
            fringes, positions, deviation_cm, path_length_cm)
    else:
        wavelength, wavelength_uncertainty, slope, std_err, r_squared = calculate_wavelength(
            fringes, positions)
    
    # 计算不确定度
    fringes_range = np.max(fringes) - np.min(fringes)
    total_uncertainty = calculate_uncertainty(slope, std_err, fringes_range)
    
    # 绘制图表
    fig = plot_data_and_fit(
        fringes, positions, slope, intercept=0, 
        wavelength=wavelength, wavelength_uncertainty=wavelength_uncertainty,
        r_squared=r_squared, total_uncertainty=total_uncertainty
    )
    
    # 将图表转换为base64编码
    fig_base64 = fig_to_base64(fig)
    
    # 整理结果
    results = {
        "wavelength": wavelength,
        "wavelength_uncertainty": wavelength_uncertainty,
        "total_uncertainty": total_uncertainty,
        "r_squared": r_squared,
        "slope": slope,
        "std_err": std_err
    }
    
    return results, fig_base64

def generate_html(results, fig_base64):
    """
    生成分析结果的HTML页面
    
    参数:
    results -- 包含分析结果的字典
    fig_base64 -- base64编码的图表
    
    返回:
    html -- HTML页面内容
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>迈克尔逊干涉实验分析结果</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            .result-box {{ border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 20px; }}
            .result-title {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
            .result-item {{ margin-bottom: 5px; }}
            .plot-container {{ text-align: center; margin: 20px 0; }}
            .plot-img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>迈克尔逊干涉实验分析结果</h1>
            
            <div class="result-box">
                <div class="result-title">测量结果</div>
                <div class="result-item"><strong>波长:</strong> {results["wavelength"]:.2f} ± {results["wavelength_uncertainty"]:.2f} nm</div>
                <div class="result-item"><strong>相对不确定度:</strong> {results["total_uncertainty"]:.2f}%</div>
                <div class="result-item"><strong>拟合优度 R²:</strong> {results["r_squared"]:.6f}</div>
                <div class="result-item"><strong>斜率:</strong> {results["slope"]:.8f} mm/圈</div>
                <div class="result-item"><strong>斜率标准不确定度:</strong> {results["std_err"]:.8f} mm/圈</div>
            </div>
            
            <div class="plot-container">
                <img class="plot-img" src="data:image/png;base64,{fig_base64}" alt="数据拟合图">
            </div>
            
            <div class="result-box">
                <div class="result-title">实验建议</div>
                <div class="result-item">1. 确保实验光路调整良好，尽量减小条纹中心偏移</div>
                <div class="result-item">2. 校准刻度后，建议沿同一方向继续旋转微调鼓轮3-4圈后再开始记录数据，以消除螺纹空程差</div>
                <div class="result-item">3. 在条纹比较稀疏(镜面间距较小)时调节条纹中心到理想位置</div>
                <div class="result-item">4. 使用更多的数据点可以提高测量精度</div>
            </div>
        </div>
    </body>
    </html>
    """
    return html

def create_online_version():
    """创建可以在线使用的HTML表单页面"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>迈克尔逊干涉实验数据分析</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            h1 { text-align: center; color: #2c3e50; }
            .form-group { margin-bottom: 15px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input[type="text"], textarea, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
            button { background-color: #3498db; color: white; border: none; padding: 10px 15px; border-radius: 4px; cursor: pointer; font-size: 16px; }
            button:hover { background-color: #2980b9; }
            .input-method { margin-bottom: 20px; }
            .data-input { display: none; }
            .instructions { background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
            .tab-container { display: flex; margin-bottom: 20px; }
            .tab { padding: 10px 15px; background-color: #eee; cursor: pointer; border: 1px solid #ddd; border-radius: 4px 4px 0 0; margin-right: 5px; }
            .tab.active { background-color: #3498db; color: white; }
            .error { color: red; margin-top: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>迈克尔逊干涉实验数据分析</h1>
            
            <div class="instructions">
                <h3>使用说明</h3>
                <p>本工具用于分析迈克尔逊干涉实验数据，计算激光波长并进行不确定度分析。</p>
                <p>请选择数据输入方式，然后输入相应的实验数据。</p>
                <p><strong>注意：</strong> 为了减小系统不确定度，建议在刻度校准后沿同一方向继续旋转微调鼓轮3-4圈后再开始记录数据。</p>
            </div>
            
            <div class="tab-container">
                <div class="tab active" id="tab-pair">一对一输入</div>
                <div class="tab" id="tab-bulk">批量输入</div>
            </div>
            
            <div class="data-input" id="pair-input" style="display: block;">
                <div class="form-group">
                    <label>请输入数据 (每行输入一组"圈数 位置"，如"0 0.09212"):</label>
                    <textarea id="pair-data" rows="10" placeholder="0 0.09212
50 0.10865
100 0.12514
..."></textarea>
                    <div id="pair-error" class="error"></div>
                </div>
            </div>
            
            <div class="data-input" id="bulk-input">
                <div class="form-group">
                    <label>请输入圈数数据 (用空格分隔):</label>
                    <input type="text" id="fringes-data" placeholder="0 50 100 150 200 250 300 350 400 450 500 550">
                    <div id="fringes-error" class="error"></div>
                </div>
                
                <div class="form-group">
                    <label>请输入对应的位置数据 (用空格分隔，单位mm):</label>
                    <input type="text" id="positions-data" placeholder="54.410 54.4275 54.4435 54.4591 54.4749 54.4908 54.5059 54.5217 54.5374 54.5529 54.5687 54.5845">
                    <div id="positions-error" class="error"></div>
                </div>
            </div>
            
            <div class="form-group">
                <label>高级选项:</label>
                <div>
                    <input type="checkbox" id="correct-backlash" checked>
                    <label for="correct-backlash" style="display: inline;">校正螺纹空程差</label>
                </div>
                
                <div style="margin-top: 10px;">
                    <label for="deviation">条纹中心偏移量 (cm):</label>
                    <input type="text" id="deviation" value="0" style="width: 100px;">
                </div>
                
                <div style="margin-top: 10px;">
                    <label for="path-length">S1到毛玻璃屏的距离 (cm):</label>
                    <input type="text" id="path-length" value="41" style="width: 100px;">
                </div>
            </div>
            
            <div style="text-align: center; margin-top: 20px;">
                <button id="analyze-btn">分析数据</button>
            </div>
            
            <div id="results-container" style="margin-top: 30px; display: none;"></div>
            
            <script>
                // 切换输入方式
                document.getElementById('tab-pair').addEventListener('click', function() {
                    document.getElementById('tab-pair').classList.add('active');
                    document.getElementById('tab-bulk').classList.remove('active');
                    document.getElementById('pair-input').style.display = 'block';
                    document.getElementById('bulk-input').style.display = 'none';
                });
                
                document.getElementById('tab-bulk').addEventListener('click', function() {
                    document.getElementById('tab-bulk').classList.add('active');
                    document.getElementById('tab-pair').classList.remove('active');
                    document.getElementById('bulk-input').style.display = 'block';
                    document.getElementById('pair-input').style.display = 'none';
                });
                
                // 数据分析按钮事件
                document.getElementById('analyze-btn').addEventListener('click', function() {
                    let fringes = [];
                    let positions = [];
                    let validData = true;
                    
                    // 清除所有错误提示
                    document.querySelectorAll('.error').forEach(el => el.textContent = '');
                    
                    if (document.getElementById('pair-input').style.display === 'block') {
                        // 一对一输入方式
                        const pairData = document.getElementById('pair-data').value.trim();
                        if (!pairData) {
                            document.getElementById('pair-error').textContent = '请输入数据';
                            validData = false;
                        } else {
                            const lines = pairData.split('\\n');
                            for (let i = 0; i < lines.length; i++) {
                                const line = lines[i].trim();
                                if (!line) continue;
                                
                                const parts = line.split(/\\s+/);
                                if (parts.length !== 2) {
                                    document.getElementById('pair-error').textContent = `第${i+1}行格式错误，应为"圈数 位置"`;
                                    validData = false;
                                    break;
                                }
                                
                                const fringe = parseFloat(parts[0]);
                                const position = parseFloat(parts[1]);
                                
                                if (isNaN(fringe) || isNaN(position)) {
                                    document.getElementById('pair-error').textContent = `第${i+1}行包含无效数字`;
                                    validData = false;
                                    break;
                                }
                                
                                fringes.push(fringe);
                                positions.push(position);
                            }
                        }
                    } else {
                        // 批量输入方式
                        const fringesInput = document.getElementById('fringes-data').value.trim();
                        const positionsInput = document.getElementById('positions-data').value.trim();
                        
                        if (!fringesInput) {
                            document.getElementById('fringes-error').textContent = '请输入圈数数据';
                            validData = false;
                        }
                        
                        if (!positionsInput) {
                            document.getElementById('positions-error').textContent = '请输入位置数据';
                            validData = false;
                        }
                        
                        if (validData) {
                            fringes = fringesInput.split(/\\s+/).map(Number);
                            positions = positionsInput.split(/\\s+/).map(Number);
                            
                            if (fringes.some(isNaN)) {
                                document.getElementById('fringes-error').textContent = '圈数数据包含无效数字';
                                validData = false;
                            }
                            
                            if (positions.some(isNaN)) {
                                document.getElementById('positions-error').textContent = '位置数据包含无效数字';
                                validData = false;
                            }
                            
                            if (fringes.length !== positions.length) {
                                document.getElementById('positions-error').textContent = '圈数和位置的数据点数量不一致';
                                validData = false;
                            }
                        }
                    }
                    
                    if (validData && fringes.length < 2) {
                        document.getElementById('pair-error').textContent = '至少需要2个数据点进行拟合';
                        validData = false;
                    }
                    
                    if (validData) {
                        const deviation = parseFloat(document.getElementById('deviation').value) || 0;
                        const pathLength = parseFloat(document.getElementById('path-length').value) || 41;
                        const correctBacklash = document.getElementById('correct-backlash').checked;
                        
                        // 这里应当发送数据到服务器进行处理
                        // 由于这是一个静态HTML示例，我们只能显示一个假结果
                        document.getElementById('results-container').innerHTML = `
                            <div style="text-align: center; padding: 20px;">
                                <h2>分析结果</h2>
                                <p>这只是一个演示页面，需要与后端服务配合使用。</p>
                                <p>请使用完整的Python程序处理数据。</p>
                                <p>您输入了 ${fringes.length} 个数据点。</p>
                            </div>
                        `;
                        document.getElementById('results-container').style.display = 'block';
                    }
                });
            </script>
        </div>
    </body>
    </html>
    """
    return html

def process_data():
    """
    命令行交互式数据处理程序
    """
    print("=" * 50)
    print("迈克尔逊干涉实验数据处理程序 (优化版)")
    print("=" * 50)
    
    # 获取数据输入方式
    print("\n请选择数据输入方式:")
    print("1. 一对一输入圈数和位置")
    print("2. 批量输入多组数据")
    
    choice = input("请输入选择 (1 或 2): ")
    
    fringes = []
    positions = []
    
    if choice == '1':
        # 一对一输入
        print("\n请输入数据 (每行输入一组'圈数 位置'，输入空行结束):")
        print("例如：0 0.09212")
        
        while True:
            line = input().strip()
            if not line:  # 空行结束输入
                break
                
            try:
                parts = line.split()
                fringe = float(parts[0])
                position = float(parts[1])
                fringes.append(fringe)
                positions.append(position)
                print(f"已添加: 圈数={fringe}, 位置={position}mm")
            except:
                print("输入格式错误，请重新输入")
    
    elif choice == '2':
        # 批量输入
        print("\n请输入圈数数据 (用空格分隔):")
        fringes_input = input().strip()
        fringes = [float(f) for f in fringes_input.split()]
        
        print("请输入对应的位置数据 (用空格分隔)，单位mm:")
        positions_input = input().strip()
        positions = [float(p) for p in positions_input.split()]
        
        # 检查数据长度是否一致
        if len(fringes) != len(positions):
            print("错误：圈数和位置的数据点数量不一致")
            return
    
    else:
        print("无效选择，程序退出")
        return
    
    # 检查数据点数量
    if len(fringes) < 2:
        print("错误：至少需要2个数据点进行拟合")
        return
    
    print("\n输入的数据:")
    print("序号  圈数     位置(mm)")
    print("-" * 25)
    for i, (f, p) in enumerate(zip(fringes, positions)):
        print(f"{i+1:<5} {f:<8} {p:<10.6f}")
    
    # 询问是否校正螺纹空程差
    correct_backlash = input("\n是否校正螺纹空程差？(y/n，推荐选y): ").lower() == 'y'
    
    # 询问光路偏移情况
    try:
        deviation = float(input("\n请输入条纹中心偏移量(cm，如果没有请输入0): "))
    except:
        deviation = 0
    
    try:
        path_length = float(input("\n请输入S1到毛玻璃屏的距离(cm，默认41cm): ") or "41")
    except:
        path_length = 41
    
    # 分析数据
    results, _ = analyze_data(
        fringes, positions, 
        deviation_cm=deviation, 
        path_length_cm=path_length,
        correct_backlash=correct_backlash
    )
    
    # 显示结果
    print("\n=== 计算结果 ===")
    print(f"斜率 = {results['slope']:.8f} ± {results['std_err']:.8f} mm/圈")
    print(f"波长 = {results['wavelength']:.2f} ± {results['wavelength_uncertainty']:.2f} nm")
    print(f"相对不确定度 = {results['total_uncertainty']:.2f}%")
    print(f"R² = {results['r_squared']:.6f}")
    
    # 询问是否生成HTML报告
    save_html = input("\n是否生成HTML分析报告？(y/n): ").lower() == 'y'
    if save_html:
        results, fig_base64 = analyze_data(
            fringes, positions, 
            deviation_cm=deviation, 
            path_length_cm=path_length,
            correct_backlash=correct_backlash
        )
        
        html_content = generate_html(results, fig_base64)
        
        filename = input("请输入保存的文件名(默认为'michelson_analysis.html'): ")
        if not filename:
            filename = 'michelson_analysis.html'
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML报告已保存为: {filename}")
    
    # 询问是否生成在线版本
    generate_online = input("\n是否生成可分享的在线版本HTML？(y/n): ").lower() == 'y'
    if generate_online:
        online_html = create_online_version()
        
        filename = input("请输入保存的文件名(默认为'michelson_analyzer_online.html'): ")
        if not filename:
            filename = 'michelson_analyzer_online.html'
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(online_html)
        
        print(f"在线版本已保存为: {filename}")
        print("注意：此在线版本需要配合后端服务使用，或者嵌入到支持Python的环境中才能完成实际计算")
    
    # 绘制图形
    wavelength, wavelength_uncertainty, slope, std_err, r_squared = calculate_wavelength(fringes, positions)
    fringes_range = np.max(fringes) - np.min(fringes)
    total_uncertainty = calculate_uncertainty(slope, std_err, fringes_range)
    
    fig = plot_data_and_fit(
        fringes, positions, slope, intercept=0, 
        wavelength=results['wavelength'], 
        wavelength_uncertainty=results['wavelength_uncertainty'],
        r_squared=results['r_squared'],
        total_uncertainty=results['total_uncertainty']
    )
    
    # 显示图形
    plt.figure(fig.number)
    plt.show()
    
    # 询问是否保存图形
    save_choice = input("\n是否保存分析图形？(y/n): ")
    if save_choice.lower() == 'y':
        filename = input("请输入保存的文件名(默认为'michelson_analysis.png'): ")
        if not filename:
            filename = 'michelson_analysis.png'
        fig.savefig(filename, dpi=300)
        print(f"图形已保存为: {filename}")

    print("\n处理完成!")

if __name__ == "__main__":
    process_data()