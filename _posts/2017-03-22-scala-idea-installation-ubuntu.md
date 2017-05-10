---
title: install scala and idea on ubuntu 16.04
categories:
  - technical posts
tags:
  - scala
  - idea
date: 2017-03-22 12:24:30 +0000
---

## 安装scala

sudo apt-get install scala

## 用ubuntu-make安装idea

不推荐在ubuntu 16.04上直接用apt-get安装的ubuntu-make.

### 安装ubuntu-make

sudo add-apt-repository ppa:ubuntu-desktop/ubuntu-make
sudo apt-get update
sudo apt-get install ubuntu-make

### 安装idea

umake ide idea


## 直接用ppa安装idea

sudo add-apt-repository ppa:mmk2410/intellij-idea-community
sudo apt-get update
sudo apt-get install intellij-idea-community


## 启动idea

我的安装直接在远程的ubuntu虚拟机上.所有直接启动idea.sh

建立别名

alias idea='$HOME/.local/share/umake/ide/idea/bin/idea.sh'
idea

## 安装scala插件

启动后需要先进行configure而不是创建新项目,选择configure->plugins,然后选择install JetBrains Plugins, 然后找到scala插件点击install.

## Hello world

可以创建新项目, 选择项目类型为scala, 项目名称为HelloWorld, 在创建新项目时可能需要指定jdk和scala的sdk

![Create New Project]({{ site.url}}/post-images/scala-idea-installation-ubuntu/scala-idea-installation-ubuntu-01.PNG)

在系统中用which和'ls -l'找到javac和scala的安装目录即可

项目创建后在src目录上右键单击选择新建scala class,名称为HelloWorld, kind为Object

![Create New Class]({{ site.url}}/post-images/scala-idea-installation-ubuntu/scala-idea-installation-ubuntu-02.PNG)

然后输入下列代码

```
object HelloWorld {

  def main(args: Array[String]): Unit = {
    println("Hello, world!")
  }

}
```

右键选择HelloWorld->Run HelloWold即可

![Run HelloWorld]({{ site.url}}/post-images/scala-idea-installation-ubuntu/scala-idea-installation-ubuntu-03.PNG)

## 参考

https://itsfoss.com/install-intellij-ubuntu-linux/
