## Prepare
```bash
conda create -n diffusion python=3.9
conda activate diffusion
pip install -r requirements.txt
```

## Train
```bash
python train.py
```

# Play
```bash
python play.py
```

# analysis
noise_pred_net:
1. 输入：
    * `noised_action`: 噪声化的动作 (shape: batch_size × pred_horizon × action_dim)
    * `timestep`: 扩散迭代步数
    * `global_cond`: 全局条件，这里是展平的观测值 (从 obs_horizon × obs_dim 展平)
2. 网络架构：
    * `下采样路径`: 通过多层下采样模块减小特征图尺寸
    * `中间层`: 处理下采样后的特征
    * `上采样路径`: 通过上采样模块恢复特征图尺寸，并结合跳跃连接
    * `最终输出层`: 将特征映射回原始动作维度，预测添加的噪声
3. 条件机制：
    * 扩散步骤编码: 通过 `diffusion_step_encoder` 编码当前扩散迭代步数
    * 全局条件: 通过 FiLM (Feature-wise Linear Modulation) 将观测信息整合到网络中
4. 处理流程
    * `timestep`: 首先通过`diffusion_step_encoder`进行编码，转换为高维特征向量，编码过程SinusoidalPosEmb → Linear → Mish → Linear
    * 结合`global_cond`: 编码后的`timestep`特征与`global_cond`（即展平的观测值）沿最后一个维度连接，形成一个统一的条件向量：`global_feature = torch.cat([global_feature, global_cond], axis=-1)`
    * 处理`noised_action`: `noised_action` 首先被重排维度，从 (B,T,C) 转换为 (B,C,T)，以适应卷积操作，然后作为网络的主输入 x 进入 U-Net 的下采样路径
    * 条件应用机制：在每个`ConditionalResidualBlock1D`中，条件信息`global_feature`通过`FiLM`机制应用：`out = scale * out + bias`

训练:
1. 读取数据集里的`nobs`和`naction`
2. 随机产生高斯噪声
3. 随机采样timesteps (属于[0, `noise_scheduler.config.num_train_timesteps`])，添加噪声`noisy_actions=noise_scheduler.add_noise(naction, noise, timesteps)`
4.使用`noise_pred_net`预测噪声`noise_pred = noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)`
5. 计算noise loss `mse_loss(noise_pred, noise)`，反向传播
6. Update Exponential Moving Average of the model weights: `ema.step(noise_pred_net.parameters())`

部署:
1. 读取`nobs`
2. 随机产生高斯噪声`noisy_action`(维度是`action_dim`)
3. 预测噪声`noise_pred_net(sample=naction, timestep=k, global_cond=obs_cond)`
4. 去噪`naction=noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample`
5. 只执行前`action_horizon`个动作

