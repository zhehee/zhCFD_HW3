import numpy as np
import matplotlib.pyplot as plt

L = 3.0  # 空间域长度
T = 2.0  # 总模拟时间
nu = 1.0  # 波速

def upwind_solver(u, nt, dx, dt):
    """一阶迎风格式"""
    sigma = nu * dt / dx
    for _ in range(nt):
        u_new = u.copy()
        u_new[1:] = u[1:] - sigma * (u[1:] - u[:-1])
        u_new[0] = u[0] - sigma * (u[0] - u[-1])  # 周期边界
        u = u_new
    return u


def lax_wendroff_solver(u, nt, dx, dt):
    """Lax-Wendroff格式 (二阶)"""
    sigma = nu * dt / dx
    for _ in range(nt):
        u_new = u.copy()

        # 内部节点
        u_new[1:-1] = u[1:-1] - sigma * (u[2:] - u[:-2]) / 2 + sigma **2 * (u[2:] - 2 * u[1:-1] + u[:-2]) / 2
        # 周期边界处理
        u_new[0] = u[0] - sigma * (u[1] - u[-1]) / 2 + sigma **2 * (u[1] - 2 * u[0] + u[-1]) / 2
        u_new[-1] = u[-1] - sigma * (u[0] - u[-2]) / 2 + sigma **2 * (u[0] - 2 * u[-1] + u[-2]) / 2
        u = u_new
    return u


def warming_beam_solver(u, nt, dx, dt):
    """Warming-Beam格式 (三阶迎风)"""
    sigma = nu * dt / dx
    for _ in range(nt):
        u_new = u.copy()
        # 内部节点 (三阶离散)
        u_new[2:] = u[2:] -  sigma * (u[2:] - u[1:-1]) - 0.5 * sigma * (1 - sigma) * (u[2:] - 2 * u[1:-1] + u[:-2])
        # 边界处理 (一阶迎风)
        u_new[0] = u[0] - sigma * (u[0] - u[-1]) - 0.5 * sigma * (1 - sigma) * (u[0] - 2 * u[-1] + u[-2])
        u_new[1] = u[1] - sigma * (u[1] - u[0]) - 0.5 * sigma * (1 - sigma) * (u[1] - 2 * u[0] + u[-1])# 周期边界
        u = u_new
    return u



# 精确解
def exact_solution(x, t):
    return np.sin(2 * np.pi * (x - nu * t))



# 稳定性验证
def validate_stability():
    schemes = [
        {'name': 'upwind', 'stable_sigma': 0.9, 'unstable_sigma': 1.1},
        {'name': 'lax_wendroff', 'stable_sigma': 0.9, 'unstable_sigma': 1.1},
        {'name': 'warming_beam', 'stable_sigma': 1.8, 'unstable_sigma': 2.2}
    ]

    nx = 300
    dx = L / nx
    x = np.linspace(0, L, nx)
    u_initial = np.sin(2 * np.pi * x )

    for scheme in schemes:
        plt.figure(figsize=(10, 4))

        # 计算稳定和不稳定解
        for case, sigma in [('stable solution', scheme['stable_sigma']),
                            ('unstable solution', scheme['unstable_sigma'])]:
            dt = sigma * dx / nu
            nt = int(T / dt)

            # 选择求解器
            if scheme['name'] == 'upwind':
                u_num = upwind_solver(u_initial.copy(), nt, dx, dt)
            elif scheme['name'] == 'lax_wendroff':
                u_num = lax_wendroff_solver(u_initial.copy(), nt, dx, dt)
            elif scheme['name'] == 'warming_beam':
                u_num = warming_beam_solver(u_initial.copy(), nt, dx, dt)

            # 精确解计算
            t_final = nt * dt
            u_exact = exact_solution(x, t_final)

            # 绘制结果
            plt.plot(x, u_num,
                     linestyle='-' if case == 'stable solution' else ':',
                     linewidth=2,
                     label=f'{case} (σ={sigma})')
            plt.plot(x, u_exact, 'k--', alpha=0.5, label='exact solution')

        plt.ylim(-2, 2)  # 扩大坐标范围显示不稳定震荡
        plt.title(f"{scheme['name'].capitalize()} comparison of format stability ")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        # 添加PDF保存
        plt.savefig(f"{scheme['name']}_stability.pdf", bbox_inches='tight', dpi=300)
        plt.close()
    plt.show()
# 精度验证
def validate_convergence():
    plt.figure(figsize=(10, 6))
    nx_list = [150, 300, 600, 1200, 2400]  # 细化网格
    markers = ['o', 's', 'D', '^', 'v']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    schemes = ['upwind', 'lax_wendroff', 'warming_beam']

    # 定义各格式的CFL数
    sigma_dict = {
        'upwind': 0.8,  # σ < 1
        'lax_wendroff': 0.8,  # σ < 1
        'warming_beam': 1.8  # σ < 2
    }

    for idx, scheme in enumerate(schemes):
        errors = []
        dx_list = []
        sigma = sigma_dict[scheme]

        for nx in nx_list:
            dx = L / nx
            dt = sigma * dx / nu  # 保持CFL数恒定
            nt = int(T / dt)
            x = np.linspace(0, L, nx, endpoint=False)

            u_initial = np.sin(2 * np.pi * x)

            # 计算数值解
            if scheme == 'upwind':
                u_num = upwind_solver(u_initial.copy(), nt, dx, dt)
            elif scheme == 'lax_wendroff':
                u_num = lax_wendroff_solver(u_initial.copy(), nt, dx, dt)
            elif scheme == 'warming_beam':
                u_num = warming_beam_solver(u_initial.copy(), nt, dx, dt)

            # 精确解 (波长为1)
            t_final = nt * dt
            x_exact = (x - nu * t_final) % L  # 周期边界处理
            u_exact = np.sin(2 * np.pi * x_exact)  # 波长=1

            # 计算L2误差
            error = np.sqrt(np.mean((u_num - u_exact) ** 2))
            errors.append(error)
            dx_list.append(dx)

        # 收敛阶拟合 (使用后三个网格点)
        log_dx = np.log10(dx_list[-3:])
        log_err = np.log10(errors[-3:])
        coeffs = np.polyfit(log_dx, log_err, 1)

        plt.plot(log_dx, log_err,
                 marker=markers[idx], color=colors[idx], linestyle='--',
                 label=f'{scheme} (slope={coeffs[0]:.2f})')

    plt.xlabel(r'$\log_{10}(\Delta x)$', fontsize=12)
    plt.ylabel(r'$\log_{10}(\mathrm{L2\ Error})$', fontsize=12)
    plt.title('Convergence Analysis (Wavelength=1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("convergence_analysis.pdf", bbox_inches='tight', dpi=300)
    plt.close()


# 耗散与相位分析

def analyze_dissipation_phase():
    L = 3.0
    nu = 1.0
    T = 2.0
    nx = 600
    dx = L / nx
    x = np.linspace(0, L, nx)
    u_initial = np.sin(2 * np.pi * x)

    # 定义各格式的σ范围 (根据稳定性条件)
    schemes = [
        {'name': 'Upwind', 'solver': upwind_solver, 'sigma_range': np.linspace(0.1, 0.95, 50)},  # σ < 1
        {'name': 'Lax-Wendroff', 'solver': lax_wendroff_solver, 'sigma_range': np.linspace(0.1, 0.95, 50)},  # σ < 1
        {'name': 'Warming-Beam', 'solver': warming_beam_solver, 'sigma_range': np.linspace(0.1, 1.9, 50)}  # σ < 2
    ]

    # 绘制振幅衰减曲线
    plt.figure(figsize=(8, 6))
    for scheme in schemes:
        amp_losses = []
        valid_sigmas = []
        for sigma in scheme['sigma_range']:
            try:
                dt = sigma * dx / nu
                nt = int(T / dt)
                u_num = scheme['solver'](u_initial.copy(), nt, dx, dt)
                amplitude = (u_num.max() - u_num.min()) / 2
                amp_loss = 1 - amplitude
                amp_losses.append(amp_loss)
                valid_sigmas.append(sigma)
            except:
                continue  # 跳过不稳定的σ
        plt.plot(valid_sigmas, amp_losses, 'o-', label=scheme['name'])
    plt.xlabel(r'CFL number $\sigma$')
    plt.ylabel('Amplitude Damping Factor')
    plt.title('Amplitude Damping vs. CFL Number')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("amplitude_damping.pdf", bbox_inches='tight', dpi=300)
    plt.close()

    # 绘制相位误差曲线
    plt.figure(figsize=(8, 6))
    for scheme in schemes:
        phase_errors = []
        valid_sigmas = []
        for sigma in scheme['sigma_range']:
            try:
                dt = sigma * dx / nu
                nt = int(T / dt)
                u_num = scheme['solver'](u_initial.copy(), nt, dx, dt)
                u_exact = exact_solution(x, nt * dt)

                # 计算相位差
                peak_num = x[np.argmax(u_num)]
                x_exact_peak = x[np.argmax(u_exact)]
                phase_diff = (peak_num - x_exact_peak) % 1
                if phase_diff > 0.5:
                    phase_diff = 1 - phase_diff
                phase_errors.append(phase_diff)
                valid_sigmas.append(sigma)
            except:
                continue
        plt.plot(valid_sigmas, phase_errors, 'o-', label=scheme['name'])
    plt.xlabel(r'CFL number $\sigma$')
    plt.ylabel('Phase Error (Fraction of Wavelength)')
    plt.title('Phase Error vs. CFL Number')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("phase_error.pdf", bbox_inches='tight', dpi=300)
    plt.close()

# 执行所有验证

validate_stability()
validate_convergence()
analyze_dissipation_phase()
plt.show()