import sys
import os
sys.path.append(os.getcwd())

from src.env.environment import Environment, LEAD_PROGRAM_SIMPLE, LEAD_PROGRAM_COMPLEX, LEAD_PROGRAM_RANDOMISH
from src.controller.actor_critic_controller import ActorCriticController

from src import frontend


def main():
    # create environment e
    # run e on new process
    env = Environment(1200.0, 800.0, 15.0, 7.0, LEAD_PROGRAM_RANDOMISH)
    env.run()

    ActorCriticController.run(
        ActorCriticController,
        env.following_cart,
        env.lead_cart,
        1200,
        bootstrap_policy=True,
        bootstrap_policy_loss_threshold=10.0,
        policy_network_save_file_path=r"D:\bazyli\Dropbox\code\PythonProjects\lead-follow\model-saves\policy.pt",
        pretrain_critic=True,
        pretrain_critic_loss_threshold=0.01,
        value_network_save_file_path=r"D:\bazyli\Dropbox\code\PythonProjects\lead-follow\model-saves\critic.pt")

    # create frontend / main window
    # and start its event loop
    frontend.create_and_run(env.create_gfx_items(), 1200, 800)


if __name__ == "__main__":
    main()
