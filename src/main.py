import sys
import os
sys.path.append(os.getcwd())

from src.env.environment import Environment
from src.controller.pid_controller import PIDController, PIDControllerFactory
from src import frontend


def main():
    # create environment e
    # run e on new process
    env = Environment(1200.0, 800.0, 15.0, 7.0)
    env.run()

    PIDController.run(PIDControllerFactory.create, env.following_cart, env.lead_cart)

    # create frontend / main window
    # and start its event loop
    frontend.create_and_run(env.create_gfx_items(), 1200, 800)


if __name__ == "__main__":
    main()
