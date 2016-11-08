---
title: python flask 简介
categories:
  - technical posts
tags:
  - python
  - flask
date: 2016-11-07 12:24:30 +0000
---

# Flask 简介

Flask 是一个基于Werkzeug, Jinja 2的微框架, BSD License

Werkzeug 是一个 WSGI 工具包, WSGI 是一个 Web 应用和服务器通信的协议，Web 应用 可以通过 WSGI 一起工作.
Jinja2 是一个现代的，设计者友好的，仿照 Django 模板的 Python 模板语言

## 安装

```
$ flask/bin/pip install flask
$ flask/bin/pip install flask-login
$ flask/bin/pip install flask-openid
$ flask/bin/pip install flask-mail
$ flask/bin/pip install flask-sqlalchemy
$ flask/bin/pip install sqlalchemy-migrate
$ flask/bin/pip install flask-whooshalchemy
$ flask/bin/pip install flask-wtf
$ flask/bin/pip install flask-babel
$ flask/bin/pip install guess_language
$ flask/bin/pip install flipflop
$ flask/bin/pip install coverage
```

## 第一个app

建立第一个app, 仅仅是返回'Hello World!'  

```
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

if __name__ == '__main__':
    app.run()
```

启动app  

```
python hello.py
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

测试一下,另开一个终端

```
curl http://127.0.0.1:5000/
```

说明

* 我们导入了 Flask 类。这个类的实例将会是我们的 WSGI 应用程序
* 我们创建一个该类的实例，第一个参数是应用模块或者包的名称。 如果你使用单一的模块（如本例），你应该使用 __name__ ，因为模块的名称将会因其作为单独应用启动还是作为模块导入而有不同（ 也即是 '__main__' 或实际的导入名）。这是必须的，这样 Flask 才知道到哪去找模板、静态文件等等
* 我们使用 route() 装饰器告诉 Flask 什么样的URL 能触发我们的函数
* 这个函数的名字也在生成 URL 时被特定的函数采用，这个函数返回我们想要显示在用户浏览器中的信息。
* 最后我们用 run() 函数来让应用运行在本地服务器上

## 改进

### 外部可访问的服务器  

```
app.run(host='0.0.0.0')
```

### 调试模式

```
app.run(debug=True)
```

## 路由

### 基本用法

route() 装饰器把一个函数绑定到对应的URL上

```
@app.route('/')
def index():
    return 'Index Page'

@app.route('/hello')
def hello():
    return 'Hello World'
```    

用下面的命令在两个终端上试试

```
python hello.py

curl localhost:5000
curl localhost:5000/hello

```

### 在URL中使用变量

```
@app.route('/user/<username>')
def show_user_profile(username):
    # show the user profile for that user
    return 'User %s' % username

@app.route('/post/<int:post_id>')
def show_post(post_id):
    # show the post with the given id, the id is an integer
    return 'Post %d' % post_id
```

输入的变量可以指定转换器将字符串型的变量转为int或float. 也可以转成path型变量接受斜杠

### 规范 URL

```
@app.route('/projects/')
```
访问/projects时会被重定向到/projects/, 即/projects和/projects/都工作


```
@app.route('/about')
```
访问/about/时不工作.

### 构造 URL

Flask 能匹配URL,Flask也可以生成它们. 当Flask回传的页面中有url时,很明显Flask需要有能构造URL的能力

```python
from flask import Flask, url_for
app = Flask(__name__)
@app.route('/')
def index(): pass

@app.route('/login')
def login(): pass

@app.route('/user/<username>')
def profile(username): pass

with app.test_request_context():
  print url_for('index')
  print url_for('login')
  print url_for('login', next='/')
  print url_for('profile', username='John Doe')
```

为什么你要构建 URL 而非在模板中硬编码？这里有三个绝妙的理由

* 反向构建通常比硬编码的描述性更好。更重要的是，它允许你一次性修改 URL,而不是到处边找边改.
* URL 构建会转义特殊字符和 Unicode 数据,免去你很多麻烦.
* 如果你的应用不位于 URL 的根路径()比如,在 /myapplication 下,而不是 /)url_for() 会妥善处理这个问题.

### HTTP 方法

HTTP(与 Web 应用会话的协议)有许多不同的访问URL方法. 默认情况下,路由只回应 GET 请求,但是通过 route() 装饰器传递 methods 参数可以改变这个行为

```
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        do_the_login()
    else:
        show_the_login_form()
```

* GET: 浏览器告知服务器. 只获取页面上的信息并发给我.这是最常用的方法.
* HEAD: 浏览器告诉服务器. 欲获取信息,但是只关心消息头.应用应像处理GET请求一样来处理它,但是不分发实际内容.在 Flask中你完全无需人工干预,底层的Werkzeug库已经替你打点好了.
POST: 浏览器告诉服务器. 想在URL上发布新信息.并且,服务器必须确保数据已存储且仅存储一次.这是HTML表单通常发送数据到服务器的方法.
* PUT: 类似POST但是服务器可能触发了存储过程多次,多次覆盖掉旧值.你可能会问这有什么用,当然这是有原因的.考虑到传输中连接可能会丢失,在这种情况下浏览器和服务器之间的系统可能安全地第二次接收请求,而不破坏其它东西.因为POST它只触发一次.
* DELETE: 删除给定位置的信息.
* OPTIONS: 给客户端提供一个敏捷的途径来弄清这个 URL支持哪些HTTP方法。 从Flask 0.6开始,实现了自动处理.

有趣的是,在HTML4和XHTML1中,表单只能以GET和POST方法提交到服务器.但是 JavaScript 和未来的 HTML 标准允许你使用其它所有的方法.此外,HTTP 最近变得相当流行,浏览器不再是唯一的 HTTP 客户端.比如,许多版本控制系统就在使用 HTTP.

## 静态文件

动态 web 应用也会需要静态文件,通常是CSS和JavaScript文件.理想状况下,你已经配置好Web服务器来提供静态文件,但是在开发中,Flask也可以做到.只要在你的包中或是模块的所在目录中创建一个名为static的文件夹,在应用中使用/static 即可访问.

```
url_for('static', filename='style.css')
```

## 模板

### 模板渲染

用Python生成HTML十分无趣,而且相当繁琐,因为你必须手动对HTML做转义来保证应用的安全.为此,Flask配备了Jinja2模板引擎.

你可以使用render_template()方法来渲染模板.你需要做的一切就是将模板名和你想作为关键字的参数传入模板的变量.这里有一个展示如何渲染模板的简例:

```
from flask import render_template

@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', name=name)
```

Flask会在templates文件夹里寻找模板.如果你的应用是个模块,这个文件夹应该与模块同级;如果它是一个包,那么这个文件夹作为包的子目录

* 应用是个模块

```
/application.py
/templates
    /hello.html
```

* 应用是个包

```
/application
    /__init__.py
    /templates
        /hello.html
```

hello.html的模板例子

```
{% raw %}
<!doctype html>
<title>Hello from Flask</title>
{% if name %}
  <h1>Hello {{ name }}!</h1>
{% else %}
  <h1>Hello World!</h1>
{% endif %}
{% endraw %}
```

在模板里,你也可以访问request, session和g[1]对象, 以及get_flashed_messages()函数

g对象是一个可以存储应用上下文的对象

### 模板继承

Jinja 最为强大的地方在于他的模板继承功能,模板继承允许你创建一个基础的骨架模板, 这个模板包含您网站的通用元素,并且定义子模板可以重载的 blocks.

定义一个基础模板,base.html

```
{% raw %}
<html>
  <head>
    {% if title %}
    <title>{{title}} - microblog</title>
    {% else %}
    <title>microblog</title>
    {% endif %}
  </head>
  <body>
    <div>Microblog: <a href="/index">Home</a></div>
    <hr>
    {% block content %}{% endblock %}
  </body>
</html>
{% endraw %}
```

建立index.html 模板继承自 base.html

```
{% raw %}
{% extends "base.html" %}
{% block content %}
<h1>Hi, {{user.nickname}}!</h1>
{% for post in posts %}
<div><p>{{post.author.nickname}} says: <b>{{post.body}}</b></p></div>
{% endfor %}
{% endblock %}
{% endraw %}
```

使用继承的模板 view.py

```
def index():
    user = { 'nickname': 'Miguel' } # fake user
    posts = [ # fake array of posts
        {
            'author': { 'nickname': 'John' },
            'body': 'Beautiful day in Portland!'
        },
        {
            'author': { 'nickname': 'Susan' },
            'body': 'The Avengers movie was so cool!'
        }
    ]
    return render_template("index.html",
        title = 'Home',
        user = user,
        posts = posts)
```

## 对象

### 请求对象

HTTP方法可通过request对象的method属性来访问

```
from flask import request

@app.route('/login', methods=['POST', 'GET'])
def login():
    error = None
    if request.method == 'POST':
        if valid_login(request.form['username'],
                       request.form['password']):
            return log_the_user_in(request.form['username'])
        else:
            error = 'Invalid username/password'
    # the code below is executed if the request method
    # was GET or the credentials were invalid
    return render_template('login.html', error=error)
```

通过request对象的args属性来访问URL中提交的参数(?key=value)

```
searchword = request.args.get('q', '')
```

### 文件上传

首先,在HTML表单中设置enctype="multipart/form-data"属性

已上传的文件存储在内存或是文件系统中一个临时的位置.你可以通过请求对象的files属性访问它们.每个上传的文件都会存储在这个字典里

```
from flask import request

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['the_file']
        f.save('/var/www/uploads/uploaded_file.txt')
    ...
```

如果你想知道上传前文件在客户端的文件名是什么,你可以访问filename属性. 这个值是可以伪造的.如果你要把文件按客户端提供的文件名存储在服务器上,那么请把它传递给Werkzeug提供的secure_filename()函数

```
from flask import request
from werkzeug import secure_filename

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['the_file']
        f.save('/var/www/uploads/' + secure_filename(f.filename))
    ...
```

### Cookies

你可以通过cookies属性来访问Cookies,用响应对象的set_cookie方法来设置Cookies. 请求对象的cookies属性是一个内容为客户端提交的所有 Cookies 的字典.如果你想使用会话,请不要直接使用 Cookies

读取 cookies:

```
from flask import request

@app.route('/')
def index():
    username = request.cookies.get('username')
    # use cookies.get(key) instead of cookies[key] to not get a
    # KeyError if the cookie is missing.
```

存储 cookies:

```
from flask import make_response

@app.route('/')
def index():
    resp = make_response(render_template(...))
    resp.set_cookie('username', 'the username')
    return resp
```

Cookies 是设置在响应对象上的.由于通常视图函数只是返回字符串,之后 Flask 将字符串转换为响应对象.如果你要显式地转换,你可以使用 make_response() 函数然后再进行修改.

## 重定向和错误

redirect和abort

```
from flask import abort, redirect, url_for

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login')
def login():
    abort(401)
    this_is_never_executed()
```

## 会话

session对象允许你在不同请求间存储特定用户的信息.它是在 Cookies 的基础上实现的,并且对 Cookies 进行密钥签名.这意味着用户可以查看你 Cookie 的内容,但却不能修改它,除非用户知道签名的密钥.

要使用会话,你需要设置一个密钥.这里介绍会话如何工作

```
from flask import Flask, session, redirect, url_for, escape, request

app = Flask(__name__)

@app.route('/')
def index():
    if 'username' in session:
        return 'Logged in as %s' % escape(session['username'])
    return 'You are not logged in'

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['username'] = request.form['username']
        return redirect(url_for('index'))
    return '''
        <form action="" method="post">
            <p><input type=text name=username>
            <p><input type=submit value=Login>
        </form>
    '''

@app.route('/logout')
def logout():
    # remove the username from the session if it's there
    session.pop('username', None)
    return redirect(url_for('index'))

# set the secret key.  keep this really secret:
app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'
```

## 日志记录

这里有一些调用日志记录的例子:

```
app.logger.debug('A value for debugging')
app.logger.warning('A warning occurred (%d apples)', 42)
app.logger.error('An error occurred')
```

## 总结

上面的例子基本来自网上的教程. 基本上我对Flask应用的需求还是去实现某些支持Restful接口的后台应用, 接受http请求的方法和参数. 返回json或yml格式的响应. 如果要建完全的网站,可能还是去考虑Django或者Pyramid

## 参考

http://www.pythondoc.com/flask-mega-tutorial/
http://docs.jinkan.org/docs/flask/
http://werkzeug-docs-cn.readthedocs.io/zh_CN/latest/index.html
http://docs.jinkan.org/docs/jinja2/
