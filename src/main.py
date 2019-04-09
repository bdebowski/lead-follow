from src.env.environment import Environment
from src.backend import Backend
from src.frontend import Frontend


def main():
    # create environment e
    # run e on new process
    env = Environment()
    backend = Backend(env)
    backend.run()

    # create frontend / main window
    # and start its event loop
    frontend = Frontend()
    frontend.run()


if __name__ == "__main__":
    main()
