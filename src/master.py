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


def main(simple=False):
    # ensure current working directory is in src folder
    if os.getcwd()[-3:] != 'src':
        # assuming we are somewhere inside the git directory
        path = s.Popen('git rev-parse --show-toplevel', shell=True, stdout=s.PIPE).communicate()[0].decode("utf-8")[:-1]
        print('changing working directory from', os.getcwd(), 'to', path)
        os.chdir(path + '/src')

    if simple:  # simple refers to limited parameters fed into run.py, 2 refers to
        s.call('python3 run.py master True '+str(2)+' False', shell=True)
        copy_contents("funding")
        return

    s.call('mkdir ../zipped_archives/master', shell=True)

    key = ['Bayesian', "Heuristic", "DNN"]
    folder_list = []
    count = 1
    has_big_data = False
    for alg in [2, 3, 4, 4]:
        print("\n\n------------------ALG {}------------------\n\n".format(alg))
        s.call('mkdir ../zipped_archives/master/'+key[alg-2]+str(count), shell=True)
        params = str(alg) + ' ' + str(has_big_data)

        print("\n\n------------------FUNDING------------------\n\n")
        s.call('python3 run.py master True '+params, shell=True)
        copy_contents("funding")
        save_contents(key[alg-2]+str(count), "funding")

        print("\n\n------------------NO FUNDING------------------\n\n")
        s.call('python3 run.py master False '+params, shell=True)
        copy_contents("no_funding")
        save_contents(key[alg-2]+str(count), "no_funding")

        collect()
        save_data(key[alg-2]+str(count))
        folder_list.append(key[alg-2]+str(count))
        has_big_data = True  # all future runs add on to big data

        if (alg == 4 and count == 1) or alg == 3:
            print("\n\n------------------TRAINING DNN------------------\n\n")
            s.call('python3 ../ai/neural_net.py', shell=True)  # training the neural network now

        if alg == 4:
            count = 2

    generate_html(folder_list)


def collect(simple=True):
    if simple:
        print("\n\n------------------COLLECTING------------------\n\n")
        s.call('python collect.py master', shell=True)
    else:
        key = ['Bayesian', "Heuristic", "DNN"]
        count = 1
        for alg in [2, 3, 4, 4]:
            print("\n\n------------------ALG {}------------------\n\n".format(alg))
            move_contents(key[alg - 2] + str(count))
            collect()
            save_data(key[alg - 2] + str(count))

            if alg == 4:
                count = 2


def move_contents(folder):
    s.call('cp -r ../zipped_archives/master/'+folder+'/funding ../zipped_archives/master/', shell=True)
    s.call('cp -r ../zipped_archives/master/'+folder+'/no_funding ../zipped_archives/master/', shell=True)


def save_data(folder):
    # saving the graphs/data
    s.call('mkdir ../zipped_archives/master/'+folder+'/data', shell=True)
    s.call('cp -r ../data/ ../zipped_archives/master/'+folder+'/data', shell=True)


def save_contents(folder, name):
    s.call('mkdir ../zipped_archives/master/'+folder+'/'+name, shell=True)
    s.call('cp -r tmp/model/ ../zipped_archives/master/'+folder+'/'+name, shell=True)


def generate_html(folder_list):
    html = ''
    for i in folder_list:
        html += '<a href="{0}/data/pages/all_images.html">{0}</a><br />'.format(i)
    with open('../zipped_archives/master/image_explorer.html', 'w') as file:
        file.write(html)


if __name__ == '__main__':
    main()
    # collect()
    collect(simple=False)
    # main(simple=True)
