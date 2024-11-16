import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
import argparse
from collections import deque
from scipy.spatial.transform import Rotation as R
from legged_gym import LEGGED_GYM_ROOT_DIR

from legged_gym.envs.go1.go1_config import Go1RoughCfg
import torch

asset_dof_names = ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
                   'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
                   'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
                   'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']

default_dof_pos = np.zeros(12)
for i in range(len(asset_dof_names)):
    default_dof_pos[i] = Go1RoughCfg.init_state.default_joint_angles[asset_dof_names[i]]

class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0

def low_pass_action_filter(actions, last_actions):
  alpha = 0.2
  actons_filtered = last_actions * alpha + actions * (1 - alpha)
  return actons_filtered

def get_obs(data):
    # q = data.qpos.astype(np.double)
    # dq = data.qvel.astype(np.double)
    # q = q[-cfg.env.num_actions:]
    # dq = dq[-cfg.env.num_actions:]
    q = np.zeros(12)
    dq = np.zeros(12)
    for i in range(12):
        q[i] = data.joint(asset_dof_names[i]).qpos[0]
        dq[i] = data.joint(asset_dof_names[i]).qvel[0]

    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

def run_mujoco(policy, cfg):
    global default_dof_pos
    model = mujoco.MjModel.from_xml_path(cfg.mjsim_config.mujoco_model_path) # 载入初始化位置由XML决定
    model.opt.timestep = cfg.mjsim_config.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)
    command = np.zeros((1, 3), dtype=np.double)
    last_action = action[:]
    hist_obs_vec = np.zeros([1, cfg.env.history_length * cfg.env.num_one_step_observations], dtype=np.double)
    
    hist_obs_deque = deque()
    for _ in range(cfg.env.history_length):
        hist_obs_deque.append(np.zeros([1, cfg.env.num_one_step_observations], dtype=np.double))
    
    count_lowlevel = 0

    for _ in tqdm(range(int(cfg.mjsim_config.sim_duration / cfg.mjsim_config.dt)), desc="Simulating..."):

        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data) #从mujoco获取仿真数据
        if True:
            # 1000hz ->50hz
            if count_lowlevel % cfg.mjsim_config.decimation == 0:
                obs = np.zeros([1, cfg.env.num_one_step_observations], dtype=np.double) # 1,45
                # ---- set command ------
                cmd.vx = 0.7
                cmd.vy = 0.0 
                cmd.dyaw = 0.0
                # ---- set command ------
                
                # current_obs = torch.cat((self.commands[:, :3] * self.commands_scale,
                #             self.base_ang_vel  * self.obs_scales.ang_vel,
                #             self.projected_gravity,
                #             (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                #             self.dof_vel * self.obs_scales.dof_vel,
                #             self.actions
                #             ),dim=-1)
                
                obs[0, 0] = cmd.vx * cfg.normalization.obs_scales.lin_vel
                obs[0, 1] = cmd.vy * cfg.normalization.obs_scales.lin_vel
                obs[0, 2] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
                obs[0, 3] = omega[0] * cfg.normalization.obs_scales.ang_vel
                obs[0, 4] = omega[1] * cfg.normalization.obs_scales.ang_vel
                obs[0, 5] = omega[2] * cfg.normalization.obs_scales.ang_vel
                
                obs[0, 6] = v[0] * cfg.normalization.obs_scales.lin_vel
                obs[0, 7] = v[1] * cfg.normalization.obs_scales.lin_vel
                obs[0, 8] = v[2] * cfg.normalization.obs_scales.lin_vel
                
                obs[0, 9] = gvec[0]
                obs[0, 10] = gvec[1]
                obs[0, 11] = gvec[2]
                obs[0, 12:24] = (q - default_dof_pos) * cfg.normalization.obs_scales.dof_pos
                obs[0, 24:36] = dq * cfg.normalization.obs_scales.dof_vel
                obs[0, 36:48] = last_action
                
                hist_obs_deque.append(obs)
                hist_obs_deque.popleft()

                #-------缓存历史观测作为模型输入---
                for i in range(cfg.env.history_length):#缓存历史观测
                    hist_obs_vec[0, i * cfg.env.num_one_step_observations : (i + 1) * cfg.env.num_one_step_observations] = hist_obs_deque[cfg.env.history_length - 1 - i][0, :]
                
                #-------推理jit save的模型-------
                policy = policy.to('cpu')
                action[:] = policy(torch.tensor(hist_obs_vec).to(torch.float32))[0].detach().numpy()#jit模型

                #-------计算力矩输出-------------
                action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
                action_flt = 0.8 * action + 0.2 * last_action
                last_action = action
                #-------计算前馈力矩-------------
                actions_scaled = action_flt * cfg.control.action_scale
                actions_scaled[[0, 3, 6, 9]] *= cfg.control.hip_reduction
                target_q = actions_scaled + default_dof_pos 
            
            if cfg.control.control_type=="P":
                torques = (target_q - q) * cfg.robot_config.kps - dq * cfg.robot_config.kds
            else:
                raise NameError(f"Unknown controller type: {cfg.control.control_type}")
            
            # torques = np.clip (torques, - cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
            data.ctrl = torques
        else: # air mode test
            target_q = default_dof_pos
            target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
            # Generate PD control
            tau = (target_q - q) * cfg.robot_config.kps + (target_dq - dq) * cfg.robot_config.kds
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
            data.ctrl = tau # np.zeros((cfg.env.num_actions), dtype=np.double)
            
            # for i in range(sim.data.ncon):
            #     con = sim.data.contact[i]
            #     if (con.geom1 == geom1_id and con.geom2 == geom2_id) or (con.geom2 == geom1_id and con.geom1 == geom2_id):
            #         contact_pos = con.pos
            #         break
            
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
            body_pos = data.xpos[body_id]
            print(f"The position of the body base is: {body_pos}")

        mujoco.mj_step(model, data)
        if count_lowlevel % cfg.mjsim_config.decimation == 0:
            # viewer.cam.distance = 3.0
            # viewer.cam.azimuth = 90
            # viewer.cam.elevation = -45
            viewer.cam.lookat[:] = data.qpos.astype(np.double)[0:3]
            viewer.render()
        count_lowlevel += 1

    viewer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, default='model_vqvae.pt',
                        help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', default=False)
    args = parser.parse_args()
    
    args.terrain = True
    args.load_model = f'{LEGGED_GYM_ROOT_DIR}/logs/rough_go1/exported/policies/policy.pt'
    
    class Sim2simCfg(Go1RoughCfg):
        class mjsim_config:
            if args.terrain:
                # mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/xml/world_terrain.xml'
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/xml/world_stairs.xml'
            else:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/xml/world.xml'
            sim_duration = 60.0
            dt = 0.001
            decimation = 20 #50Hz

        class robot_config:
            kp_all = Go1RoughCfg.control.stiffness['joint'] 
            kd_all = Go1RoughCfg.control.damping['joint'] 
            kps = np.array([kp_all, kp_all, kp_all, kp_all, kp_all, kp_all, kp_all, kp_all, kp_all, kp_all, kp_all, kp_all], dtype=np.double)
            kds = np.array([kd_all, kd_all, kd_all, kd_all, kd_all, kd_all, kd_all, kd_all, kd_all, kd_all, kd_all, kd_all], dtype=np.double)
            tau_limit = np.array([20., 55., 55., 20., 55., 55., 20., 55., 55., 20., 55., 55.])

        class env(Go1RoughCfg.env):
            pass
            
        class normalization(Go1RoughCfg.normalization):
            pass            

    policy = torch.jit.load(args.load_model)
    policy.eval() 

    run_mujoco(policy, Sim2simCfg())
