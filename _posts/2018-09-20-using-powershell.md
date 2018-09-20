---
title: using powershell
categories:
  - technical posts
tags:
  - powershell
  - windows
date: 2018-09-20 21:23:26 +0100
---


## powershell 介绍

在windows环境下,微软出的shell. 也就是说如果不想用cmd,而更喜欢使用类似linux的shell环境
的话,可以启动powershell

## 别名与shell函数

shell环境下的一个好处是可以添加好多不同的别名和shell脚本函数来快速执行一些常用命令, 例如

* 快速切换到常用工作目录
  + function CDREPOS {set-location c:\repos}
  + 创建了一个简单函数cdrepos用于切换当前目录
* 从命令行调用常用的编辑器或者其他的应用程序
  + Set-Alias -Name Edit -Value 'C:\Program Files\Notepad++\notepad++.exe'
  + 创建了别名Edit,用于调用notepad++, 也就是说用Edit today.txt就是用notepadd++来编辑文件today.txt
* 创建别名或者函数来仿造linux shell下的常用命令
  + Get-Alias ls
  + 你可以看到,powershell自动设定了别名ls(linux的常用命令),用来显示当前目录的内容
* 习惯使用某些工具的命令行界面,例如git, npm等等

## profile与脚本

别名和函数需要每次在启动powershell后添加, 每次都添加就会很繁琐, 为了方便应当把这些
设置别名和函数命令放在一个脚本文件,每次只需要执行该脚本文件就可以.

更方便的方法是,如果该脚本文件命名为一个特殊的文件, 该文件会在powershell启动时被自动执行.
该特殊文件的路径存储在变量$profile中,  可以用命令显示该文件的路径,

```
echo $profile
```

编辑该文件,并加入想要定义的内容,例如
```
Set-Alias -Name Edit -Value 'C:\Program Files\Notepad++\notepad++.exe'

function CDREPOS {set-location c:\repos}
```

每次启动后,该脚本会被自动执行. Edit命令就会启动notepad++, 而cdrepos就等效于'cd c:\repos'

## 设置ExecutionPolicy.

在前面的步骤完成后, $profile仍然不会被执行, 因为缺省的执行策略不允许自动执行脚本.
一般建议修改自动脚本执行的策略为RemoteSigned,意思是所有网络下载的脚本需要被签名,
如果签名证书从来未见过,则会提示是否信任该证书签名的脚本.

* 首先启动powershell并且run as administrator
* Set-ExecutionPolicy RemoteSigned  

## 参考
