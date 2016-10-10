---
title: jekyll quick guide
categories:
  - technical posts
tags:
  - jekyll
---
# Jekyll 快速指南

描述: 本快速指南用于在github上建立一个博客

## 第一部分 在Ubuntu系统上的安装

### 安装准备  

提示: 我现在安装的Jekyll需要Ruby 2.0以上版本，ubuntu 14.04上的ruby是1.9.3，建议简单的方式是安装ubuntu高一点的版本

sudo apt-get install ruby ruby-dev make gcc nodejs
sudo apt-get install git  
sudo apt-get install zlib

### 安装Jekyll  

```
sudo gem install bundler  
sudo gem install jekyll --no-rdoc --no-ri
```

### 检查安装的jekyll版本

jekyll -v

### 建立一个新的站点并检查jekyll是否可以运行

1. 建立一个新的站点，这会创建一个目录  
```
jekyll new my-awesome-site
```
2. 进入站点目录并启动jekyll  
```
cd my-awesome-site
jekyll serve
```
3. 通过浏览器查看
访问地址 127.0.0.1:4000

### 其他的一些有用的命令或选项  

1. 安装github-pages gem.  
提示: 这个gem绑定了github pages支持的几个其他的gems.  
```
sudo gem install github-pages --no-rdoc --no-ri
```
2. 检查站点目录中的变动并重新生成站点  
```
jekyll serve -w
```
3. 改变服务端口号  
```
jekyll serve --port 4001
```

## 第二部分 在github上设立你的站点
github内建了jekyll并提供了相应的服务叫做github-pages,github自动使用jekyll来解析你建立的站点生成相应的html文件

1. 在github上创建一个repository,取名为my_test_site, 选择为public，同时选择Initialize this repository with a README
2. 第一步只是创建了一个repository,这个站点还没有部署在github上，这一步是建立一个名为gh-pages的分支，对于我们这个演示来说master分支已经没用了，可以直接删除这个master分支
3. Clone Repository到本地  
```bash
git clone https://github.com/eyulush/my_test_site.git
```
4. 直接用jekyll建立初始的站点  
提示: 在已有的目录中建立站点而不是新建一个目录，clone下来的目录是非空的，所以用--force选项   
```
cd my_test_site  
jekyll new . --force
```
5. 配置你的站点  
jekyll的站点配置文件是站点目录下的_config.yml   
修改baseurl为/my_test_site,你也可以配置上title, url,等
6. 使用git来commit新建的站点文件  
```
git add .  
git commit -m 'first commit for new test site'
```
7. 使用git来push内容到github  
```
git push  
```
8. 在网上查看新建立的站点  

在github上找到你的repository，选择settings，因为该repository有gh-pages分支，所以提示有"Your site is published at https://eyulush.github.io/my_test_site/" 信息，点击链接就可以看到新建的站点啦

## 第三部分 使用主题和模板快速建立站点

### Jekyll的主题
Jekyll尝试去支持主题，并想支持在主题间切换，很不幸目前支持的主题只有一个就是你在前面新建站点是所用的minmia.

### Jekyll bootstrap的主题
参考链接 http://jekyllbootstrap.com/

### 克隆一个站点的repo作为模板，然后根据自己的需求来定制这个站点来形成自己的站点
这是一个通用的方法，本文将以此为基础来快速的假设站点

#### 寻找模板

在google上可以搜到很多模板和模板集合的网站  
http://jekyll.tips/templates/  

我个人比较喜欢https://mmistakes.github.io/minimal-mistakes/，下面就以minimal-mistakes为模板来建立站点

#### 安装模板
minimal-mistakes模板的安装教程在 https://mmistakes.github.io/minimal-mistakes/docs/quick-start-guide/

1. 直接克隆repo的gh-pages到本地  
```
git clone -b gh-pages https://mmistakes.github.io/minimal-mistakes.git
```  
2. 在本地进行定制修改，使用本地的jekyll进行查看  
```  
bundle install
bundle exec jekyll serve --config _config.yml _config.dev.yml  
```  
访问地址 http://localhost:4000/try_template_mm/  
3. 在github新建一个repository,并克隆到本地
说明:
现在github的规则是，如果你新建的repository的名称为USERNAME.github.io，你的站点的访问地址就是https://USERNAME.github.io  
同时这种类型的站点只能从master分支发布, 如果你repository名称为其他如XXXX, 则可以从master发布，也可以从gh-pages分支发布，在repository的settngs选择  
4. 将定制后的站点导入新建的repository，之后提交到github上

#### 定制模板

1. 修改标题等
直接编辑_config.yml文件，设置title为'鱼头的技术小站'
设定baseurl和url,下面是例子  
```
title                    : "鱼头的技术小站"
title_separator          : "-"
name                     : &name "Lucius Yu"
description              : &description "A flexible Jekyll theme for your blog or site with a minimalist aesthetic."
url                      : https://eyulush.github.io # the base hostname & protocol for your site e.g. "https://mmistakes.github.io"
baseurl                  : "/try_template_mm" # the subpath of your site, e.g. "/blog"
```

2. 设置菜单
菜单的定义文件为_data/navigation.yml

3. 设置个人头像
头像的图片放在_image目录下，同时更改_config.yml设置你的个人介绍和个人头像的图片

4. 去除广告
广告设置在_layout/目录下的default.html中,直接注释掉就好了   
```
<!-- Remove google ads
<script async src="//pagead2.googlesyndication.com/pagead/js/adsbygoogle.js"></script>
-->
```

5. 修改主页
主页的设定页面在_pages目录下的home.md
修改主页时如果只是想注释掉某些部分，在Jekyll使用的Liquid语言上注释是   
```
{% comment %}
...
{% endcomment %}
```

6. 修改关于页面
关于页面在_pages目录下的about.md

7. 修改条款与隐私页面

8. 设定License

#### 如何写文档

###### 文档格式的说明

文档直接使用markdown来编写，写好的文档放在_doc目录下，每篇文档的开头学要如下的声明部分，例如_doc目录下的07-ui-text.md  
```
---
title: "UI Text"
permalink: /docs/ui-text/
excerpt: "Text for customizing various user interface elements found in the theme."
modified: 2016-04-27T10:35:05-04:00
---  
```

说明: permlink指明在Jekyll解析到本文时，对应生成的文件路径，在上面的例子中07-ui-text.md对应生成_site/docs/ui-text/index.html文件

博客的编写
一般的post文章放在_post目录下，post文章的开头格式如下例所示  


```
---
title: "Edge Case: Nested and Mixed Lists"
categories:
  - Edge Case
tags:
  - content
  - css
  - edge case
  - lists
  - markup
---
```

说明:   
categories指明了类别，在_pages目录下有一个category-archive.html给出了所有post文章按照类别排序后的链接。同时，每个post在生成页面上带有所属类别的标签，点击标签可以跳转到生成的category-archive/index.html该类别的部分  
tags该post带有的标签，一篇post可以多个标签，用法同categories.

###### rake的简短说明
每篇文章都需要在开始出输入固定格式的内容，这比较繁琐，一个简单的办法是用个小工具来帮助自动生成文档和文章的固定格式部分就好了。可以用任何语言来做这个小工具，如shell脚本，python等等. 不过，通行的方法是用一个工具rake来做这件工作，rake就是ruby的make. 需要建立一个rakefile的文件，在文件中定义task，给出该task的依赖，和task的执行规则.
