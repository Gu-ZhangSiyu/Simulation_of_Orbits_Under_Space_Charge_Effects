#include <mpi.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <chrono>
#include <random>
#include <cstdlib>

struct Particle {
    double x, y, z;      // 位置坐标
    double vx, vy, vz;   // 速度分量
    double ax, ay, az;   // 加速度分量
    double t_inject;     // 粒子注入系统的时间
    double weight;       // 每个超粒子代表的真实粒子数（例如 1000）
    bool active;         // 判断粒子是否仍在模拟区域中, true 表示有效
    bool target_recorded; // 标靶记录标志（首次穿过标靶时记录）
};

// 辅助函数：创建 MPI 数据类型用于 Particle
void createParticleMPIType(MPI_Datatype* mpi_particle_type)
{
    // Particle 中共有 13 个成员（注意 bool 类型用 MPI_CXX_BOOL）
    const int nitems = 13;
    int blocklengths[nitems] = {1,1,1, 1,1,1, 1,1,1, 1, 1, 1, 1};
    MPI_Datatype types[nitems] = {
        MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,   // x, y, z
        MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,   // vx, vy, vz
        MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,   // ax, ay, az
        MPI_DOUBLE,                         // t_inject
        MPI_DOUBLE,                         // weight
        MPI_CXX_BOOL,                       // active
        MPI_CXX_BOOL                        // target_recorded
    };

    MPI_Aint offsets[nitems];
    Particle dummy;
    MPI_Aint base;
    MPI_Get_address(&dummy, &base);
    MPI_Get_address(&dummy.x, &offsets[0]);
    MPI_Get_address(&dummy.y, &offsets[1]);
    MPI_Get_address(&dummy.z, &offsets[2]);
    MPI_Get_address(&dummy.vx, &offsets[3]);
    MPI_Get_address(&dummy.vy, &offsets[4]);
    MPI_Get_address(&dummy.vz, &offsets[5]);
    MPI_Get_address(&dummy.ax, &offsets[6]);
    MPI_Get_address(&dummy.ay, &offsets[7]);
    MPI_Get_address(&dummy.az, &offsets[8]);
    MPI_Get_address(&dummy.t_inject, &offsets[9]);
    MPI_Get_address(&dummy.weight, &offsets[10]);
    MPI_Get_address(&dummy.active, &offsets[11]);
    MPI_Get_address(&dummy.target_recorded, &offsets[12]);

    for (int i = 0; i < nitems; i++) {
        offsets[i] = offsets[i] - base;
    }
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, mpi_particle_type);
    MPI_Type_commit(mpi_particle_type);
}

int main(int argc, char* argv[])
{
    // 初始化 MPI
    MPI_Init(&argc, &argv);
    int my_rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    MPI_Datatype mpi_particle_type;
    createParticleMPIType(&mpi_particle_type);

    using namespace std;

    // 物理常数及装置参数（与原代码一致）
    const double c = 2.99792458e8;				// 光速 (m/s)
    const double mp_MeV = 931.5;              	// 质子静质量对应能量 (MeV)
    const double q = 1.602176634e-19;        	// 基本电荷 (C)
    const double mp = 1.67262192369e-27;       	// 质子质量 (kg)
    const double eps0 = 8.854187817e-12;      	// 真空介电常数

	// 离子参数（带有4个单位正电荷，质量为11个核子单位）
    const double Z_i = 4.0;
    const double A_i = 11.0;

    // 电磁场参数及装置参数
    const double B0 = 3.0;                    	
    const double B1 = 1.6;                    	
    const double l0 = 0.7;                    	
    const double l1 = 0.5;                    	
    const double l01 = 0.5;                   	
    const double Ec = 2.5e5;                 	
    const double z_exit = 1.5;		            

    // 注入能量及时间步长参数
    const double kinj_z = 1e4;               	
    const int ndiv = 100000;                	
    const int nmax = 3400000;         			

    // 束流参数：采用超粒子方法
    // 模拟粒子数（超粒子数）
    const int num_particles_sim = 100000;
    // 真实束流中颗粒子数
    const double num_particles_real = 1e8;
    // 每个超粒子代表的真实粒子数
    double super_particle_weight = num_particles_real / num_particles_sim;

    const double beam_sigma = 0.001;			
    const double L_beam = 0.01;					
    const double z_min = -L_beam;				
    const double z_max = 0.0;					

    // 手动配置参数
    const bool enableCoulomb = true;			
    const bool enableVelocitySpread = true;		
    const double vz_sigma = 0.05; 				
    const double divergence_sigma = 1e-3; 		

    double pi = 4.0 * atan(1.0);                  		
    const double kC = 1.0 / (4 * pi * eps0);
    double q_mp = (Z_i * q) / (A_i * mp);               
    const double coulombFactor = kC * q * (q_mp);
    double omega_c = q_mp * B0;                   		
    double f_c = omega_c / (2.0 * pi);            		
    double dt = 1.0 / f_c / ndiv;                 		
    double vinj_z = sqrt(2.0 * kinj_z * q / (A_i * mp));	
    double abs_vz_sigma = vz_sigma * vinj_z;				
    double abs_div_sigma = divergence_sigma * vinj_z;		

    // 初始化随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd() + my_rank);
    std::normal_distribution<double> pos_gauss(0.0, beam_sigma);
    std::uniform_real_distribution<double> pos_uniform(z_min, z_max);
    std::normal_distribution<double> vz_dist(vinj_z, abs_vz_sigma);
    std::normal_distribution<double> divergence_dist(0.0, abs_div_sigma);

    // 由 rank 0 生成所有初始粒子，然后散发给各进程
    std::vector<Particle> particles_full;
    if(my_rank == 0)
    {
        particles_full.resize(num_particles_sim);
        for (int p = 0; p < num_particles_sim; ++p)
        {
            Particle& particle = particles_full[p];
            particle.x = pos_gauss(gen);
            particle.y = pos_gauss(gen);
            particle.z = pos_uniform(gen);
            if (enableVelocitySpread)
            {
                particle.vz = vz_dist(gen);
                particle.vx = divergence_dist(gen);
                particle.vy = divergence_dist(gen);
            }
            else
            {
                particle.vz = vinj_z;
                particle.vx = 0.0;
                particle.vy = 0.0;
            }
            particle.ax = 0.0;
            particle.ay = 0.0;
            particle.az = 0.0;
            particle.t_inject = -particle.z / vinj_z;
            particle.active = true;
            particle.target_recorded = false;
            particle.weight = super_particle_weight;  // 每个超粒子代表约 1000 个真实粒子
        }
        // 输出初始状态（initial.txt）仅由 rank 0 完成
        std::ofstream initFile("initial.txt");
        for (const auto& p : particles_full)
        {
            initFile << std::setw(22) << std::setprecision(15) << std::scientific
                     << p.x << " " << p.y << " " << p.z << "\n";
        }
        initFile.close();
        std::cout << "initial.txt output completed" << std::endl;
    }

    // 将 num_particles_sim 粒子均分给各 MPI 进程（使用 MPI_Scatterv）
    std::vector<int> counts(num_procs, 0), displs(num_procs, 0);
    int base = num_particles_sim / num_procs;
    int rem = num_particles_sim % num_procs;
    for (int i = 0; i < num_procs; i++) {
        counts[i] = base + (i < rem ? 1 : 0);
    }
    displs[0] = 0;
    for (int i = 1; i < num_procs; i++) {
        displs[i] = displs[i-1] + counts[i-1];
    }
    // 每个进程分到 local_particles 数量
    int local_num = counts[my_rank];
    std::vector<Particle> local_particles(local_num);
    
    // 使用 MPI_Scatterv 分发数据
    MPI_Scatterv(
        (my_rank==0 ? particles_full.data() : nullptr),
        counts.data(),
        displs.data(),
        mpi_particle_type,
        local_particles.data(),
        local_num,
        mpi_particle_type,
        0,
        MPI_COMM_WORLD
    );

    // 为输出准备：rank 0 写时间步数据，其他进程写带 rank 标号的文件
    std::ofstream timeFile;
    if(my_rank == 0)
        timeFile.open("t_now.txt");
    std::ostringstream spotFileName;
    spotFileName << "spot_rank" << my_rank << ".txt";
    std::ofstream spotFile(spotFileName.str());

    // 为了输出每 1000 步的全局位置数据，每个进程输出自己的数据
    
    // 主模拟循环
    double t_now = 0.0;
    auto start_time = std::chrono::high_resolution_clock::now();
    // 为全局库仑计算，每个进程在每步调用 MPI_Allgatherv 收集所有粒子
    std::vector<Particle> global_particles(num_particles_sim);

    // 为方便计算全局索引，每个进程知道自己分配的起始全局索引：
    int local_offset = displs[my_rank];

    for (int i = 1; i <= nmax; ++i)
    {
        if(my_rank == 0 && i % 1000 == 0) {
            std::cout << i << "\t" << std::fixed << std::setprecision(2)
                      << (i / static_cast<double>(nmax)) * 100 << "%" << std::endl;
        }
        t_now += dt;

        // 1. 更新外部电磁场加速度（仅对已注入粒子）
        for (int p = 0; p < local_num; ++p)
        {
            Particle& particle = local_particles[p];
            if (!particle.active || t_now < particle.t_inject) continue;
            double ec_now = 0.0, b_now = 0.0, dbdz_now = 0.0;
            if (particle.z <= l0)
            {
                ec_now = Ec;
                b_now = B0;
                dbdz_now = 0.0;
            }
            else if (particle.z <= l0 + l01)
            {
                ec_now = 0.0;
                b_now = B0 + (particle.z - l0) / l01 * (B1 - B0);
                dbdz_now = (B1 - B0) / l01;
            }
            else if (particle.z <= l0 + l01 + l1)
            {
                ec_now = 0.0;
                b_now = B1 - (particle.z - l0 - l01) / l01 * (B1 - B0);
                dbdz_now = -(B1 - B0) / l01;
            }
            else
            {
                particle.ax = 0.0;
                particle.ay = 0.0;
                particle.az = 0.0;
                continue;
            }
            // 重置加速度
            particle.ax = 0.0;
            particle.ay = 0.0;
            particle.az = 0.0;
            // 外部电磁场产生的加速度（与时间 t_now 相关）
            particle.ax += q_mp * (-ec_now * cos(omega_c * t_now) + b_now * particle.vy);
            particle.ay += q_mp * (ec_now * sin(omega_c * t_now) - b_now * particle.vx);
            particle.az += 0.5 * q_mp * (-particle.vx * dbdz_now * particle.y + particle.vy * dbdz_now * particle.x);
        }

        // 2. 计算库仑力：先全局收集所有粒子状态
        MPI_Allgatherv(
            local_particles.data(),
            local_num,
            mpi_particle_type,
            global_particles.data(),
            counts.data(),
            displs.data(),
            mpi_particle_type,
            MPI_COMM_WORLD
        );

        if (enableCoulomb)
        {
            double soft = 1e-12; // 防止除零
            // 对本地粒子累加其他粒子的库仑相互作用
            for (int iLocal = 0; iLocal < local_num; ++iLocal)
            {
                Particle& pi = local_particles[iLocal];
                if (!pi.active || t_now < pi.t_inject) continue;
                double ax_sc = 0.0, ay_sc = 0.0, az_sc = 0.0;
                int global_index = local_offset + iLocal;  // 本地粒子对应的全局索引
                for (int j = 0; j < num_particles_sim; ++j)
                {
                    // 若为同一粒子则跳过
                    if (j == global_index) continue;
                    Particle& pj = global_particles[j];
                    if (!pj.active || t_now < pj.t_inject) continue;
                    double dx = pi.x - pj.x;
                    double dy = pi.y - pj.y;
                    double dz = pi.z - pj.z;
                    double r2 = dx * dx + dy * dy + dz * dz;
                    double r = std::sqrt(r2) + soft;
                    double factor = coulombFactor / (r2 * r);
                    // 注意：超粒子方法中，每个 pj 代表 pj.weight 个真实粒子
                    ax_sc += pj.weight * factor * dx;
                    ay_sc += pj.weight * factor * dy;
                    az_sc += pj.weight * factor * dz;
                }
                pi.ax += ax_sc;
                pi.ay += ay_sc;
                pi.az += az_sc;
            }
        }

        // 3. 更新速度和位置；同时记录通过标靶的粒子
        for (int p = 0; p < local_num; ++p)
        {
            Particle& p_obj = local_particles[p];
            if (!p_obj.active || t_now < p_obj.t_inject) continue;
            p_obj.vx += p_obj.ax * dt;
            p_obj.vy += p_obj.ay * dt;
            p_obj.vz += p_obj.az * dt;
            p_obj.x += p_obj.vx * dt;
            p_obj.y += p_obj.vy * dt;
            p_obj.z += p_obj.vz * dt;
            if (!p_obj.target_recorded && p_obj.z >= z_exit)
            {
                spotFile << std::setw(22) << std::setprecision(15) << std::scientific
                         << p_obj.x << " " << p_obj.y << " " << p_obj.z << "\n";
                p_obj.target_recorded = true;
            }
        }

        // 4. 每 1000 步输出部分数据
        if (i % 1000 == 0)
        {
            // 记录第一个入射粒子轨道：简单起见，这里选 rank 0 的第一个粒子
            if (my_rank == 0 && !local_particles.empty())
            {
                Particle& pf = local_particles[0];
                // 计算动能（非相对论，单位 MeV）
                double e_first = 0.5 * mp_MeV * ((pf.vx / c) * (pf.vx / c)
                    + (pf.vy / c) * (pf.vy / c) + (pf.vz / c) * (pf.vz / c));
                timeFile << std::setprecision(15) << std::scientific
                         << t_now << " " << pf.x << " " << pf.y << " " << pf.z << " " << e_first << "\n";
                timeFile.flush();
            }
            // 每 1000 步输出本进程所有粒子位置
            std::stringstream filename;
            filename << "output_rank" << my_rank << "/" << i << ".txt";
            // 创建目录
            std::ofstream outFile(filename.str().c_str());
            if (!outFile)
            {
                std::cerr << "Error: Failed to open " << filename.str() << " for writing." << std::endl;
                exit(EXIT_FAILURE);
            }
            for (const auto& p : local_particles)
            {
                outFile << std::setw(22) << std::setprecision(15) << std::scientific
                        << p.x << " " << p.y << " " << p.z << "\n";
            }
            outFile.close();
        }
    }
    // 主循环结束

    if(my_rank == 0)
        timeFile.close();
    spotFile.close();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    if(my_rank == 0)
        std::cout << "run time: " << elapsed.count() << " s" << std::endl;

    MPI_Type_free(&mpi_particle_type);
    MPI_Finalize();
    return 0;
}
