from legged_gym import LEGGED_GYM_ROOT_DIR

from legged_gym.envs import Go1RoughCfg

import mujoco, mujoco_viewer
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

import numpy as np
import torch
import time

asset_dof_names = [
    'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
]


class cmd:
  vx = 0.0
  vy = 0.0
  dyaw = 0.0


def low_pass_action_filter(actions, last_actions):
  alpha = 0.2
  actons_filtered = last_actions * alpha + actions * (1 - alpha)
  return actons_filtered


def get_obs(data):
  q = np.zeros(12)
  dq = np.zeros(12)
  for i in range(12):
    q[i] = data.joint(asset_dof_names[i]).qpos[0]
    dq[i] = data.joint(asset_dof_names[i]).qvel[0]

  quat = data.sensor('Body_Quat').data[[1, 2, 3, 0]].astype(np.float32)
  r = R.from_quat(quat)
  v = r.apply(data.qvel[:3], inverse=True).astype(np.float32)  # In the base frame
  # print(v)
  omega = data.sensor('Body_Gyro').data.astype(np.float32)
  gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.float32)
  return (q, dq, quat, v, omega, gvec)


def playMujoco(policy, cfg):
  # load mujoco viewer
  model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
  model.opt.timestep = cfg.sim_config.dt
  data = mujoco.MjData(model)
  mujoco.mj_step(model, data)
  viewer = mujoco_viewer.MujocoViewer(model, data)

  # initial pos
  default_dof_pos = np.zeros(12)
  for i in range(len(asset_dof_names)):
    default_dof_pos[i] = cfg.init_state.default_joint_angles[asset_dof_names[i]]

  # buffers for history info
  last_action = np.zeros((cfg.env.num_actions), dtype=np.double)
  obs_buf = np.zeros([1, cfg.env.num_observations])

  # simulate demo
  count_lowlevel = 0  # low level controller frequency counter
  for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
    loop_start_time = time.time()  # record loop start time

    q, dq, quat, v, omega, gvec = get_obs(data)

    if count_lowlevel % cfg.sim_config.decimation == 0:  #
      obs = np.zeros([1, cfg.env.num_one_step_observations], dtype=np.float32)

      # command
      cmd.vx = 1.5
      cmd.vy = 0.0
      cmd.dyaw = 0.0

      # observation from simulator
      obs[0, 0] = cmd.vx * cfg.normalization.obs_scales.lin_vel
      obs[0, 1] = cmd.vy * cfg.normalization.obs_scales.lin_vel
      obs[0, 2] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
      obs[0, 3] = omega[0] * cfg.normalization.obs_scales.ang_vel
      obs[0, 4] = omega[1] * cfg.normalization.obs_scales.ang_vel
      obs[0, 5] = omega[2] * cfg.normalization.obs_scales.ang_vel
      obs[0, 6] = gvec[0]
      obs[0, 7] = gvec[1]
      obs[0, 8] = gvec[2]
      obs[0, 9:21] = (q - default_dof_pos) * cfg.normalization.obs_scales.dof_pos
      obs[0, 21:33] = dq * cfg.normalization.obs_scales.dof_vel
      obs[0, 33:45] = last_action

      # add current obs to buffer
      obs_buf = np.concatenate((obs, obs_buf[:, :-cfg.env.num_one_step_observations]), axis=-1).astype(np.float32)

      # clip actions
      action = np.clip(policy(torch.tensor(obs_buf))[0].detach().numpy(), -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
      if cfg.sim_config.use_filter:
        action_filtered = low_pass_action_filter(action, last_action)
      last_action = action

      actions_scaled = action * cfg.control.action_scale
      actions_scaled[[0, 3, 6, 9]] *= cfg.control.hip_reduction
      joint_pos_target = actions_scaled + default_dof_pos
    
    #pd controller
    p_gains = np.array([cfg.control.stiffness["joint"]] * cfg.env.num_actions)
    d_gains = np.array([cfg.control.damping["joint"]] * cfg.env.num_actions)
    if cfg.control.control_type == "P":
      torques = p_gains * (joint_pos_target - q) - d_gains * dq
    else:  # only pos control allowed
      raise NameError(f"Unknown controller type: {cfg.control.control_type}")

    data.ctrl = np.clip(torques, -cfg.sim_config.tau_limit, cfg.sim_config.tau_limit)

    # take simulate step
    mujoco.mj_step(model, data)

    time.sleep(0.001)
    if count_lowlevel % cfg.sim_config.decimation == 0:
      viewer.cam.lookat[:] = data.qpos.astype(np.float32)[0:3]
      viewer.render()
    count_lowlevel += 1

    # loop_end_time = time.time()
    # loop_elapsed_time = loop_end_time - loop_start_time
    # loop_sleep_time = cfg.sim_config.dt - loop_elapsed_time
    # if loop_sleep_time > 0:
    #   print(loop_sleep_time)
    #   time.sleep(loop_sleep_time)

  viewer.close()


if __name__ == '__main__':

  class Sim2simCfg(Go1RoughCfg):

    class sim_config:
      mujoco_model_path = "../../resources/robots/go1/xml/go1.xml"
      sim_duration = 60.0
      dt = 0.001
      decimation = 20

      use_filter = False
      tau_limit = np.array([20., 30., 30., 20., 30., 30., 20., 30., 30., 20., 30., 30.])

  policy = torch.jit.load("../../logs/rough_go1/exported/policies/policy.pt")
  policy.eval()
  policy = policy.to('cpu')

  playMujoco(policy, Sim2simCfg())
