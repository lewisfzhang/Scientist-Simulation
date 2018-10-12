# master.py
# controls iterations of algorithms & funding/no funding
# stores all of the files together, then graphs them

import subprocess as s
import os


def copy_contents(name):
    s.call('mkdir ../zipped_archives/master/'+name, shell=True)
    # s.call('cp -r ../data/pages/ ../zipped_archives/master/'+name+'/pages', shell=True)
    # s.call('cp -r ../data/images/ ../zipped_archives/master/'+name+'/pages', shell=True)
    s.call('cp -r tmp/model/ ../zipped_archives/master/'+name, shell=True)  # overwrites existing files


def main():
    # ensure current working directory is in src folder
    if os.getcwd()[-3:] != 'src':
        # assuming we are somewhere inside the git directory
        path = s.Popen('git rev-parse --show-toplevel', shell=True, stdout=s.PIPE).communicate()[0].decode("utf-8")[:-1]
        print('changing working directory from', os.getcwd(), 'to', path)
        os.chdir(path + '/src')

    s.call('mkdir ../zipped_archives/master', shell=True)

    print("\n\n------------------FUNDING------------------\n\n")
    s.call('python3 run.py master True', shell=True)
    copy_contents("funding")

    print("\n\n------------------NO FUNDING------------------\n\n")
    s.call('python3 run.py master False', shell=True)
    copy_contents("no_funding")

    # # RUN HEURISTIC
    # s.call('python3 run.py alg H', shell=True)
    # copy_contents('heuristic')
    #
    # # RUN BAYESIAN
    # s.call('python3 run.py alg B', shell=True)
    # copy_contents('bayesian')
    #
    # # RUN DNN
    # s.call('python3 run.py', shell=True)


def collect():
    print("\n\n------------------COLLECTING------------------\n\n")
    s.call('python collect.py master', shell=True)


if __name__ == '__main__':
    # main()
    collect()
