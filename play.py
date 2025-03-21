import numpy as np
import torch
import collections, os, gdown
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
from dp_utils.env import PushTEnv
from dp_utils.dataset import (
    dataset,
    pred_horizon,
    obs_horizon,
    action_horizon,
    stats,
    device,
    obs_dim,
    action_dim,
)
from dp_utils.network import ConditionalUnet1D
from dp_utils.utils import normalize_data, unnormalize_data
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import time

seed = 300

# Set up matplotlib for interactive plotting
plt.ion()

noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim, global_cond_dim=obs_dim * obs_horizon
)
load_pretrained = True
ema_noise_pred_net = noise_pred_net
if load_pretrained:
    ckpt_path = "pusht_state_100ep.ckpt"
    if not os.path.isfile(ckpt_path):
        id = "1mHDr_DEZSdiGo9yecL50BBQYzR8Fjhl_&confirm=t"
        gdown.download(id=id, output=ckpt_path, quiet=False)

    state_dict = torch.load(ckpt_path, map_location="cpu")
    ema_noise_pred_net = noise_pred_net
    ema_noise_pred_net.load_state_dict(state_dict)
    noise_pred_net.to(device)
    print("Pretrained weights loaded.")
else:
    print("Skipped pretrained weight loading.")

num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choice of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule="squaredcos_cap_v2",
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type="epsilon",
)

# limit environment interaction to 200 steps before termination
max_steps = 200
env = PushTEnv()
# use a seed >200 to avoid initial states seen in the training dataset
env.seed(300)


# get first observation
obs, info = env.reset()

# keep a queue of last 2 steps of observations
obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
# save visualization and rewards
imgs = [env.render(mode="rgb_array")]
rewards = list()
done = False
step_idx = 0

# Set up the matplotlib figure and axis for visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
img_plot = ax1.imshow(imgs[0])
ax1.set_title("Environment")
ax1.axis("off")

# Set up rewards plot
(rewards_plot,) = ax2.plot([], [], "b-")
ax2.set_title("Rewards")
ax2.set_xlabel("Steps")
ax2.set_ylabel("Reward")
ax2.grid(True)

# Add a replay button
button_ax = plt.axes([0.45, 0.01, 0.1, 0.05])
replay_button = Button(button_ax, "Replay (R)")
replay_active = False

# Initialize rewards data for plotting
reward_data = []
step_data = []


# Function to handle key press events
def on_key_press(event):
    global replay_active
    if event.key == "r" or event.key == "R":
        replay_active = True
        print("Replay activated via key press")
        replay_simulation()


# Function to handle button click
def on_button_click(event):
    global replay_active
    replay_active = True
    print("Replay activated via button click")
    replay_simulation()


# Connect the event handlers
replay_button.on_clicked(on_button_click)
fig.canvas.mpl_connect("key_press_event", on_key_press)


# Function to replay the simulation
def replay_simulation():
    # Check if we have data to replay
    if len(imgs) <= 1 or len(reward_data) <= 1:
        print("Not enough data to replay")
        return

    print(f"Replaying simulation with {len(imgs)} frames...")

    # Temporarily turn off interactive mode
    plt.ioff()

    # Reset plots
    ax1.clear()
    ax1.set_title("Environment (Replay)")
    ax1.axis("off")

    ax2.clear()
    ax2.set_title("Rewards (Replay)")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Reward")
    ax2.grid(True)

    # Create new image plot
    img_plot = ax1.imshow(imgs[0])
    (rewards_plot,) = ax2.plot([], [], "b-")

    # Begin replay
    for i in range(len(imgs)):
        # Update image
        img_plot.set_data(imgs[i])

        # Update rewards plot up to current step
        current_steps = step_data[: min(i + 1, len(step_data))]
        current_rewards = reward_data[: min(i + 1, len(reward_data))]
        rewards_plot.set_data(current_steps, current_rewards)
        ax2.relim()
        ax2.autoscale_view()

        # Display current frame number and reward
        if i < len(reward_data):
            ax1.set_title(
                f"Environment (Replay) - Frame {i}, Reward: {reward_data[i]:.4f}"
            )

        # Draw and pause
        fig.canvas.draw()
        plt.pause(0.05)  # Slightly slower than original for better visualization

    # After replay is done, show final state
    print("Replay complete")

    # Turn interactive mode back on
    plt.ion()


with tqdm(total=max_steps, desc="Eval PushTStateEnv") as pbar:
    while not done:
        B = 1
        # stack the last obs_horizon (2) number of observations
        obs_seq = np.stack(obs_deque)
        # normalize observation
        nobs = normalize_data(obs_seq, stats=stats["obs"])
        # device transfer
        nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

        # infer action
        with torch.no_grad():
            # reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

            # initialize action from Gaussian noise
            noisy_action = torch.randn((B, pred_horizon, action_dim), device=device)
            naction = noisy_action

            # init scheduler
            noise_scheduler.set_timesteps(num_diffusion_iters)

            for k in noise_scheduler.timesteps:
                # predict noise
                noise_pred = ema_noise_pred_net(
                    sample=naction, timestep=k, global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=naction
                ).prev_sample

        # unnormalize action
        naction = naction.detach().to("cpu").numpy()
        # (B, pred_horizon, action_dim)
        naction = naction[0]
        action_pred = unnormalize_data(naction, stats=stats["action"])

        # only take action_horizon number of actions
        start = obs_horizon - 1
        end = start + action_horizon
        action = action_pred[start:end, :]
        # (action_horizon, action_dim)

        # execute action_horizon number of steps
        # without replanning
        for i in range(len(action)):
            # stepping env
            obs, reward, done, _, info = env.step(action[i])
            # save observations
            obs_deque.append(obs)
            # and reward/vis
            rewards.append(reward)
            imgs.append(env.render(mode="rgb_array"))

            # Update reward data for plotting
            reward_data.append(reward)
            step_data.append(step_idx)

            # Update plots
            img_plot.set_data(imgs[-1])
            rewards_plot.set_data(step_data, reward_data)
            ax2.relim()
            ax2.autoscale_view()

            # Redraw figure
            fig.canvas.draw()
            fig.canvas.flush_events()

            # Add a small pause to allow the plots to update
            plt.pause(0.01)

            # update progress bar
            step_idx += 1
            pbar.update(1)
            pbar.set_postfix(reward=reward)
            if step_idx > max_steps:
                done = True
            if done:
                break

# print out the maximum target coverage
print("Score: ", max(rewards))

# After simulation is complete, add a message about replay functionality
print(
    "\nSimulation complete. Press 'R' key or click the 'Replay' button to replay the simulation."
)

# Keep the plot open after completion and wait for replay command
plt.ioff()

# Create message text on the figure
fig.text(
    0.5,
    0.95,
    "Simulation complete. Press 'R' key or click 'Replay' button to replay.",
    ha="center",
    va="center",
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"),
)

# Show the figure and keep it open
plt.show(block=True)

# Optional: Save the final plot if needed
fig.savefig("final_state_and_rewards.png", dpi=300, bbox_inches="tight")


# Optional: Create an animation from the collected images
def create_animation():
    ani = animation.ArtistAnimation(
        fig, [[img_plot.set_data(img)] for img in imgs], interval=50, blit=True
    )
    return ani


# Uncomment to save animation
# ani = create_animation()
# ani.save('animation.mp4', writer='ffmpeg', fps=30)
