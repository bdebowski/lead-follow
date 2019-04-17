from src.env.environment import Environment
from src.backend import Backend
from src.frontend import Frontend


def main():
    # create environment e
    # run e on new process
    env = Environment(1200.0, 800.0, 15.0, 7.0)
    backend = Backend([env])
    backend.run()

    # create frontend / main window
    # and start its event loop
    frontend = Frontend(env.create_gfx_items(), 1200, 800)
    frontend.run()


if __name__ == "__main__":
    main()
