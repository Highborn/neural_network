import sys

from src.heb import heb_neural_network


def heb():
    heb_neural_network()

if __name__ == '__main__':
    argv = sys.argv
    available_function_list = [
        heb,
    ]
    function_names = ""
    for function in available_function_list:
        function_names += "\t{}\n".format(function.__name__)
    man_msg = "Possible commands are [\n{command_names}]".format(command_names=function_names)

    if len(argv) < 2:
        print("Empty command")
        print(man_msg)
        exit(-1)

    _, command, *arguments = argv
    command = str(command).strip()

    for function in available_function_list:
        if command == function.__name__:
            function(*arguments)
            exit(0)
    else:
        print("Wrong command:\t" + command)
        print(man_msg)
