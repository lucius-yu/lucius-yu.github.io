---
title: pypiserver tutorial
categories:
  - technical post
tags:
  - python
  - pypiserver
date: 2016-05-22 21:23:26 +0100
---

## 安装pypiserver

pip install pypiserver                ## Or: pypiserver[passlib,watchdog]
mkdir ~/packages                      ## Copy packages into this directory.

手动copy python的安装包到~/packages目录下以提供给其他客户端下载,安装python的包

## 启动
pypi-server -p 8080 ~/packages &      ## Will listen to all IPs.

## 使用客户端pip来下载,安装自定义的pypiserver里提供的包

pip install  --extra-index-url http://localhost:8080/simple/ ...
pip search --index http://localhost:8080/simple/ ...

## 支持客户端上传安装包到pypiserver

### 服务端配置

#### 安装passlib包

pip install passlib

#### 每个用户创建一个用户名和密码,
下列命令会提示输入<some_username>的密码,并生成密码放入htpasswd.txt文件

htpasswd -sc htpasswd.txt <some_username>

#### 启动server

./pypi-server -p 8080 -P htpasswd.txt ~/packages &

### 客户端配置

编辑或创建 ~/.pypirc 文件

[distutils]
index-servers =
  pypi
  local

[pypi]
username:<your_pypi_username>
password:<your_pypi_passwd>

[local]
repository: http://localhost:8080
username: <some_username>
password: <some_passwd>


### 客户端上传

一般包源文件在git控制下时,上传包之前可以用git的tag来设定包版本

git tag --list -n
git tag -a x.y.z
git tag --list -n


创建的tag如果需要push到远程的git server需要使用命令

git push --tags

在python的包的源文件目录

python setup.py sdist upload -r local

## 参考

https://pypi.python.org/pypi/pypiserver#quickstart-installation-and-usage
