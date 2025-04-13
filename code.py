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
    x_wrapped = (x - nu * t) % L
    return np.sin(2 * np.pi * x_wrapped / L)



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
    u_initial = np.sin(2 * np.pi * x / L)

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
    plt.show()
# 精度验证
def validate_convergence():
    plt.figure(figsize=(10, 6))
    nx_list = [150, 300, 600, 1200]
    markers = ['o', 's', 'D']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    schemes = ['upwind', 'lax_wendroff', 'warming_beam']

    for idx, scheme in enumerate(schemes):
        errors = []
        dx_list = []
        for nx in nx_list:
            dx = L / nx
            sigma = 0.8
            dt = sigma * dx / nu
            nt = int(T / dt)
            x = np.linspace(0, L, nx)

            # 计算数值解
            u_initial = np.sin(2 * np.pi * x / L)
            if scheme == 'upwind':
                u_num = upwind_solver(u_initial.copy(), nt, dx, dt)
            elif scheme == 'lax_wendroff':
                u_num = lax_wendroff_solver(u_initial.copy(), nt, dx, dt)
            elif scheme == 'warming_beam':
                u_num = warming_beam_solver(u_initial.copy(), nt, dx, dt)

            # 误差计算
            u_exact = exact_solution(x, T)
            errors.append(np.max(np.abs(u_num - u_exact)))
            dx_list.append(dx)

        # 收敛阶拟合
        log_dx = np.log10(dx_list)
        log_err = np.log10(errors)
        coeffs = np.polyfit(log_dx, log_err, 1)

        # 绘制误差曲线
        plt.plot(log_dx, log_err,
                 marker=markers[idx], color=colors[idx], linestyle='--',
                 label=f'{scheme} (degree={coeffs[0]:.2f})')

    plt.xlabel(r'$\log_{10}(\Delta x)$', fontsize=12)
    plt.ylabel(r'$\log_{10}($maximum error$)$', fontsize=12)
    plt.title('analysis of linear fitting')
    plt.legend()
    plt.grid(True, alpha=0.3)



# 耗散与相位分析

def analyze_dissipation_phase():
    nx = 600
    dx = L / nx
    x = np.linspace(0, L, nx)
    u_initial = np.sin(2 * np.pi * x / L)

    # 定义格式及其参数
    schemes = [
        ('Upwind', upwind_solver, 0.8),
        ('Lax-Wendroff', lax_wendroff_solver, 0.8),
        ('Warming-Beam', warming_beam_solver, 1.6)
    ]

    for scheme_name, solver, sigma in schemes:
        # 计算时间步
        dt = sigma * dx / nu
        nt = int(T / dt)

        # 计算数值解
        u_num = solver(u_initial.copy(), nt, dx, dt)
        u_exact = exact_solution(x, nt * dt)

        # 耗散分析
        amplitude_num = (u_num.max() - u_num.min()) / 2
        amplitude_loss = 1 - amplitude_num

        # 相位分析
        peak_num = x[np.argmax(u_num)]
        x_exact_peak = x[np.argmax(u_exact)]
        phase_diff = (peak_num - x_exact_peak) / L

        print(f"\n{scheme_name} analysis result:")
        print(f"amplitude damping factor: {amplitude_loss:.4%}")
        if phase_diff > 0:
            print(f"Phase lead: {abs(phase_diff):.4%}wave length")
        elif phase_diff < 0:
            print(f"Phase lag: {abs(phase_diff):.4%}wave length")
        else:
            print("Phase synchronization")

        # 可视化
        plt.figure(figsize=(10, 4))
        plt.plot(x, u_num, 'r-', label=f'{scheme_name} (σ={sigma})')
        plt.plot(x, u_exact, 'k--', alpha=0.5, label='Exact solution')
        plt.scatter(peak_num, u_num.max(), c='r', s=80,
                    label=f'Numerical peak: {peak_num:.3f}')
        plt.scatter(x_exact_peak, 1.0, c='k', s=80, marker='x',
                    label=f'Exact peak: {x_exact_peak:.3f}')
        plt.title(f"Dissipation and Dispersion Analysis - {scheme_name}")
        plt.xlabel("Position")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    plt.show()

# 执行所有验证

validate_stability()
validate_convergence()
analyze_dissipation_phase()
plt.show()