---
title: python pypiserver introduction
categories:
  - technical posts
tags:
  - python
date: 2017-03-14 12:24:30 +0000
---

# python 软件包服务器pypiserver使用的简单说明

## 服务器端安装和使用说明

### Installation.
pip install pypiserver                ## Or: pypiserver[passlib,watchdog]
mkdir ~/packages                      ## Copy packages into this directory.

### Start server.

pypi-server -p 8080 ~/packages &      ## Will listen to all IPs.

如果使用password保护则

pypi-server -p 8080 -P htpasswd.txt ~/packages &

而Apached的htpasswd.txt的生成可以用下面的命令

htpasswd -sc htpasswd.txt <some_username>

## 客户端使用说明

### 客户端配置

在用户主目录下编辑配置文件~/.pypirc
```
[distutils]
index-servers =
  local

[local]
  repository: http://10.68.32.205:8080
  username: ubuntu
  password: abc123
```

### 上传安装包

cd 'your package source directory'

一般来说,包的开发会使用git,在上传包之前可以打标签,这样包的版本也会得到更新

git tag --list -n
git tag -a x.y.z
git tag --list -n

上传包的命令

python setup.py sdist upload -r local

备注

在打完标签后如果包中源文件经过了修改则上传是包的版本会x.y.z其中的z会加1,并且自动加上dev的标识

### Download and Install hosted packages.

pip install  --extra-index-url http://localhost:8080/simple/ ...

extra-index-url也可以直接写在pip的配置文件pip.conf中

### Search hosted packages

pip search --index http://10.68.32.205:8080/simple/ ...


## 参考

https://pypi.python.org/pypi/pypiserver
