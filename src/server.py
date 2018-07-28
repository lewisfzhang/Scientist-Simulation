import sys

from flask import Flask, render_template, request, redirect, Response
import subprocess as s

app = Flask(__name__)


@app.route('/')
def output():
    # serve index template
    return render_template('viewer.html', name='Joe')


@app.route('/receiver', methods=['POST'])
def worker():
    # if first key in dict is "run", ensures receiver was prompted from website
    if next(iter(dict(request.form))):
        run()
        return "success"
    else:
        return "invalid request"


@app.route('/helper', methods=['POST'])
def helper():
    # return s.Popen('echo $(cd .. && pwd)', shell=True, stdout=s.PIPE).communicate()[0].decode("utf-8")[1:-1]
    return "http://127.0.0.1:8000"


def run():
    s.call('sh run.sh', shell=True)


if __name__ == '__main__':
    # run!
    s.call('open http://127.0.0.1:1234/', shell=True)

    # start servers, 8000 is for data accessing, 1234 is main server
    try:
        s.Popen('cd ../ && nohup python -m http.server 8000 --bind 127.0.0.1 &', shell=True)
        print('127.0.0.1:8000 has started running...')
    except Exception as e:
        print('127.0.0.1:8000 already running...')
    app.run('127.0.0.1', '1234')
