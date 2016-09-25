---
title: rake quick guide
categories:
  - technical posts
tags:
  - jekyll
  - rake
  - ruby
date: 2016-09-23 13:49:49 +0200
---

# Rake 使用的简单说明

在github上写博客文章时，每篇文章都需要在开始出输入固定格式的内容，这比较繁琐，一个简单的办法是用个小工具来帮助自动生成文档和文章的固定格式部分就好了。可以用任何语言来做这个小工具，如shell脚本，python等等. 不过，通行的方法是用一个工具rake来做这件工作，rake就是ruby的make. 需要建立一个rakefile的文件，在文件中定义task，给出该task的依赖，和task的执行规则。

## 设计目标

* 文本模式下的交互式设计
* 收集要创建文件的信息
* 在_posts目录下创建文件
* 在新创建的文件中插入相应的categories,tags,date等信息
* 在已有的_posts目录下的文件中查找已经用到categories和tags信息并提示用户

## 安装rake
```bash
gem install rake
```

## 建立一个rakefile文件，建立查询已用的categories和tags

```ruby
require 'yaml'

task :get_cateories_tags do
  files=Dir["_posts/*.md"]
  files.each do |file|
    file_info = YAML.load_file(file)
    @categories.concat(file_info['categories']) if file_info['categories']
    @tags.concat(file_info['tags']) if file_info['tags']
  end
  @categories = @categories.uniq()
  @tags = @tags.uniq()
end
```
说明 :
* 在新建的markdown的文件开头部分是一段yaml语言，ruby自带yaml的解析器，只需要在rakefile开头用require请求就可以了  
* 建立一个名为get_cateories_tags的任务
* 对_posts目录下的所有的md文件用YAML解析，得到的file_info是一个哈希表，对file_info取出categories和tags的值，这个值是两个数组，分别追加到数组变量categories和tags中
* 用Array.uniq()去除重复的category和tag

## 建立收集用户信息并生成文件的task
```ruby
task :new_post do
  # fetch the existed cateories and tags
  @categories = Array.new()
  @tags = Array.new()
  Rake::Task['get_cateories_tags'].execute
  # collect the information
	puts "Input Article Title(for Article)："
	@name = STDIN.gets.chomp
	puts "Input Article Categories(#{@categories} Separated By Spaces)："
	@input_categories = STDIN.gets.chomp
  puts "Input Article Tags(#{@tags} Separated By ,)"
  @input_tags = STDIN.gets.chomp
  # generate information
  @slug = "#{@name}"
	@slug = @slug.downcase.strip.gsub(' ', '-')
	@date = Time.now.strftime("%F")
  @post_name = "_posts/#{@date}-#{@slug}.md"
  # create the new post file
  if File.exist?(@post_name)
	   abort("Failed to create the file name already exists !")
	end
	FileUtils.touch(@post_name)
  # insert the header content
  open(@post_name, 'a') do |file|
	  file.puts "---"
		file.puts "title: #{@name}"
		file.puts "categories: #{@input_categories}"
    file.puts "tags: #{@input_tags}"
    file.puts "date: #{Time.now}"
    file.puts "---"
	end
end
```

## 测试命令
```bash
rake new_post
```

## 参考链接
下面是本文参考的网上资料  
[Rake让Jekyll写博更优雅](http://www.jeffjade.com/2016/03/26/2016-03-26-rakefile-for-jekyll/)  
[Ruby 教程](http://www.runoob.com/ruby/ruby-tutorial.html)
