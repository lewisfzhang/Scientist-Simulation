# html_helper.py

import subprocess as s
import glob, init


def generate_buttons():
    template = 'var slider{0} = document.getElementById("range{0}");\n' \
               'var output{0} = document.getElementById("demo{0}");\n' \
               'output{0}.innerHTML = slider{0}.value;\n' \
               'slider{0}.oninput = function() {{\n' \
               '    output{0}.innerHTML = slider{0}.value;\n' \
               '}}\n\n'
    string = ""
    for i in range(1, 11):
        string += template.format(i)
    # print(string)
    return string


def generate_switches():
    template = '\tvar switches{0} = document.getElementById("toggle{0}");\n' \
               '\tvar s_val{0} = document.getElementById("switch{0}");\n' \
               '\ts_val{0}.innerHTML = switches{0}.checked ? "True" : "False";\n' \
               '\tswitches{0}.oninput = function() {{\n' \
               '\t\ts_val{0}.innerHTML = switches{0}.checked ? "True" : "False";\n' \
               '\t}}\n\n'
    string = ""
    for i in range(1, 5):
        string += template.format(i)
    # print(string)
    return string


def html_buttons():
    sliders = ['Time Periods', 'Ideas Per Time', 'N', 'Time Periods Alive', 'True Mean', 'SDS Prop', 'Start Effort Prop', 'K Prop', 'Switch', 'Value']
    min = [3, 1, 2, 1, 100, 0.1, 0.1, 0.1, 0, 1]
    max = [100, 100, 100, 100, 1000, 1.0, 1.0, 1.0, 3, 100]
    val = [10, 5, 10, 4, 300, 0.4, 0.5, 0.25, 2, 50]  # default settings on slider
    step = [1, 1, 1, 1, 5, 0.05, 0.05, 0.05, 1, 1]

    template = '<div class="slidecontainer container_inner">' \
               '<div class="halfcell"><p></p>' \
               '<input type="range" min="{}" max="{}" value="{}" step="{}" class="slider" id="range{}">' \
               '</div><div class="halfcell"><p align="left">{}: <span id="demo{}"></span></p></div>' \
               '<div class="halfcell"><p></p>' \
               '<input type="range" min="{}" max="{}" value="{}" step="{}" class="slider" id="range{}">' \
               '</div><div class="halfcell"><p align="left">{}: <span id="demo{}"></span></p></div>' \
               '<br><hr></div>'

    # format: (min, max, value, step, i, name, i) x 2
    out = []
    for i in range(0, len(sliders), 2):
        out.append(template.format(min[i], max[i], val[i], step[i], i+1, sliders[i], i+1,
                                   min[i+1], max[i+1], val[i+1], step[i+1], i+2, sliders[i+1], i+2))
    # print(out)
    return out


def html_switch():
    switches = ['Report All Scientists', 'Split Equal', 'Use Idea Shift', 'Show Steps']

    template = '<div class="slidecontainer container_inner">' \
               '<div class="halfcell"><p></p><label class="switch">' \
               '<input type="checkbox" id="toggle{}" checked><span class="flip"></span></label>' \
               '</div><div class="halfcell"><p align="left">{}: <span id="switch{}"></span></p></div>' \
               '<div class="halfcell"><p></p><label class="switch">' \
               '<input type="checkbox" id="toggle{}" checked><span class="flip"></span></label>' \
               '</div><div class="halfcell"><p align="left">{}: <span id="switch{}"></span></p></div>' \
               '</div>'

    # format: (i, name, i) x 2
    out = []
    for i in range(0, len(switches), 2):
        out.append(template.format(i+1, switches[i], i+1, i+2, switches[i+1], i+2))
    # print(out)
    return out


def html_slideshow():
    image_list = []
    source = "http://"+init.server_address+":8000/"
    path = init.tmp_loc + 'step/step_'
    # all step folders should have same number of images
    for image in glob.glob(path + '2/' + '*.png'):
        image_list.append(str(image)[16:])

    # format: {0} = name of image, {1} = step number / location
    head = '<div class="slideshow-container" id="{0}"><span id="span_{0}">&emsp;&emsp;&emsp;&emsp;{0}</span>'
    end = '<a class="prev" onclick="plusSlides(-1, {0})">&#10094;</a>' \
          '<a class="next" onclick="plusSlides(1, {0})">&#10095;</a>' \
          '</div>'
    filler = '<div class="mySlides{0}"><img src="{1}"></div>'

    select_head = '<p>Graph Type: <select id="chooser" onchange="displayOne()">'
    select_filler = '<option value="{0}">{1}</option>'
    select_end = '</select></p>'

    slider_head = '<input type="range" min="2" max="{}" step="1" value="2" list="steplist" id="stepper">' \
                  '<datalist id="steplist"><option selected="selected">2</option>'
    slider_filler = '<option>{}</option>'
    slider_end = '</datalist>&emsp;step: <span id="step_info"></span>'
    slider_help = '<div class="tooltip">&emsp;&emsp;&emsp;&emsp;Help?<span class="tooltiptext">' \
                  'Valid range is from step 2 to {}</span></div>'

    string1, string2, string3, good_end = '', '', '', ''
    count = 1
    steps = int(s.Popen("(cd tmp/step/ && ls -l | grep -c ^d)", shell=True,
                        stdout=s.PIPE).communicate()[0].decode('utf-8')[:-1])

    string1 += select_head
    for idx, name in enumerate(image_list):
        string1 += select_filler.format(idx, name[:-4])  # take out .png extension from name
    string1 += select_end

    string2 += slider_head.format(steps+1)  # max is inclusive, so 2 + steps - 1
    for i in range(2, steps+2):
        string2 += slider_filler.format(i)
    string2 += slider_end + slider_help.format(steps+1)

    for name in image_list:
        mid = ''
        for i in range(2, steps+2):
            mid += filler.format(count, source+'src/'+path+str(i)+'/'+name)
            good_end = end.format(count-1)  # zero based index
        count += 1
        string3 += (head + mid + good_end).format(name[:-4])  # take out .png extension from name
    # print(string1, string2, string3)
    return [len(image_list), string1, string2, string3]
