import gym
from agent import MainAgent
import plotnine as gg


env_id = "Acrobot-v1"
curr_env = gym.make(env_id)


def main():
    env_id = "Acrobot-v1"
    curr_env = gym.make(env_id)
    ppo = MainAgent(
        curr_env,
        gamma=0.99,
        lamda=0.95,
        entropy_coef=0.003,
        epsilon=0.12,
        value_range=0.5,
        timesteps=64,
        samples=4000,
        num_epochs=12,
        batch_size=32,
    )

    actor_losses, critic_losses, scores = ppo.train()

    df = actor_losses
    (
        gg.ggplot(df)
        + gg.aes(x="update_iteration", y="critic_loss")
        + gg.geom_line()
        + gg.theme_bw()
        + gg.ggtitle("Actor Loss")
    )

    df = critic_losses
    (
        gg.ggplot(df)
        + gg.aes(x="update_iteration", y="critic_loss")
        + gg.geom_line()
        + gg.theme_bw()
        + gg.ggtitle("Critic Loss")
    )

    df = scores

    (
        gg.ggplot(df)
        + gg.aes(x="df.index", y="scores")
        + gg.geom_line()
        + gg.theme_bw()
        + gg.ggtitle("Rewards")
    )


if __name__ == "__main__":
    main()
