---
title: python logging introduction
categories: 
  - technical posts
tags: 
  - python
date: 2016-10-11 12:24:30 +0000
---


# python 日志使用的简单说明

## 一个简单例子

```
import logging

# 创建一个日志句柄
logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)

# 创建一个文件日志的句柄
fh = logging.FileHandler('spam.log')
fh.setLevel(logging.DEBUG)

# 创建一个控制台输出的日志句柄
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)

# 创建一个格式化器并赋值给句柄
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# 将句柄赋值给日志对象
logger.addHandler(ch)
logger.addHandler(fh)

# 使用日志对象
logger.debug('debug message')
logger.info('info message')
logger.warn('warn message')
logger.error('error message')
logger.critical('critical message')
```

## 使用basicConfig

上面的例子中关于日志对象的配置可以使用basicConfig方法来简化,下面是第二个例子

```
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='spam.log',
                    filemode='w')

ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)

logger = logging.getLogger('simple_example2')

logger.addHandler(ch)

logger.debug('debug message')
logger.info('info message')
logger.warn('warn message')
logger.error('error message')
logger.critical('critical message')
```

上面的例子中没有创建文件日志句柄,但是缺省情况下日志中高于等于DEBUG级别的消息都会被存入到文件spam.log中.
实际上是有一个缺省句柄的配置.

而新创建的流日志句柄则是ERROR级别的,并且没有任何的formatter. 程序执行的结果就是在控制台的标准输出上为  

```bash
error message
critical message

```

在生成的日志文件spam.log中为

```
10-11 15:03 simple_example2 DEBUG    debug message
10-11 15:03 simple_example2 INFO     info message
10-11 15:03 simple_example2 WARNING  warn message
10-11 15:03 simple_example2 ERROR    error message
10-11 15:03 simple_example2 CRITICAL critical message
```

## 使用配置文件

配置文件的格式有三种, ini文件, dict式样的json文件, yaml格式的文件. 逻辑思路都是一样的只是文件的格式不同.

YAML格式的配置文件样本, logging_conf_example.yaml

```yaml
version: 1

disable_existing_loggers: False

formatters:
  simple:   
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  console:
    format: "%(name)-12s: %(levelname)-8s %(message)s"

handlers:
  console:
    class: logging.StreamHandler
    level: DEBUG
    formatter: console
    stream: ext://sys.stdout
  debugfile:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: simple
    filename: spam.log
    maxBytes: 10485760 # 10MB
    backupCount: 20
    encoding: utf8
    
loggers:
  root:
    level: DEBUG
    handlers: [debugfile,console]
    propagate: no
```

使用配置文件,需要使用yaml先载入配置文件为字典

```
import logging
import logging.config
import yaml

logging.config.dictConfig(yaml.load(open('logging_conf_example.yaml', 'r')))

logger = logging.getLogger('example')

logger.debug('debug message')
logger.info('info message')
logger.warn('warn message')
logger.error('error message')
logger.critical('critical message')

```

## 在子模块中使用logger

```
import logging

logger = logging.getLogger('example')

def foo():
    logger.info('Hi, foo')

class Bar(object):
    def bar(self):
        logger.info('Hi, bar')
```

下面是使用子模块的代码

```
import logging
import logging.config
import yaml
import module_example

logging.config.dictConfig(yaml.load(open('logging_conf_example.yaml', 'r')))

module_example.foo()
bar = module_example.Bar()
bar.bar()
```

说明 在import module_example时我们在子模块中创建了logger,但是名为example的logger还没有定义.直到使用dictConfig后.

