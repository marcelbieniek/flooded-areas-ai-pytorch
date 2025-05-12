from classification.Environment import Environment
import sys

def main():
    env = Environment()
    env.init()
    env.run()

if __name__ == "__main__":
    print(sys.path[0])
    main()
