---
title: docker quick guide
categories:
  - technical posts
tags:
  - docker
date: 2016-11-02 13:49:49 +0200
---

# Docker quick guide

## Installation

Installation: please refer to  https://docs.docker.com/engine/getstarted/linux_install_help/

Verify Installation:  

```
docker run hello-world
docker run -it ubuntu bash
docker images
docker ps -a
docker info
```

## Docker Image and Container

For the command 'docker run hello-world', it means  

* 'docker' means using docker program
* 'run' as subcommand to create and runs a container
* 'hello-world' is the image to loaded into new created container

## Build your own docker image

Following command will download image if it does not exist and then load to a new container and runs it

```
docker run docker/whalesay cowsay boo-boo
```

In this next section, the target is to improve the whalesay image by building a new version that “talks on its own” and requires fewer words to run.

### Write a Dockerfile

#### Create a new Docker file

```
mkdir mydockerbuild
cd mydockerbuild
touch Dockerfile
```

#### Edit your Docker file

* Tells Docker which image your image is based on by adding following line into Dockerfile  

```
FROM docker/whalesay:latest
```

* add the fortunes program to the image. so the program fortunes will be installed into image  

```
RUN apt-get -y update && apt-get install -y fortunes
```

* then you instruct the software to run when the image is loaded.  

```
CMD /usr/games/fortune -a | cowsay
```

* finally, the docker file should looks as Following, save and close it  

```
FROM docker/whalesay:latest
RUN apt-get -y update && apt-get install -y fortunes
CMD /usr/games/fortune -a | cowsay
```

### Build an image from your Dockerfile

build your new image by typing the 'docker build -t docker-whale .' command

### Learn about the build process

* Docker checks to make sure it has everything it needs to build  

```
Sending build context to Docker daemon 2.048 kB
```

* Then, Docker execute Step 1, 2, 3, to load with the whalesay image, install programe and finishes the build and reports its outcome.  

```
Step 1 : FROM docker/whalesay:latest
 ---> 6b362a9f73eb
Step 2 : RUN apt-get -y update && apt-get install -y fortunes
 ---> Running in 11fe2aa94d92
...
Step 3 : CMD /usr/games/fortune -a | cowsay
 ---> Running in b7e2290a9d5c
 ---> 62ea5e7f2ddd
Removing intermediate container b7e2290a9d5c
Successfully built 62ea5e7f2ddd
```

### Run your new docker-whale  

```
docker images
docker run docker-whale
```

In the example by run command 'docker run docker/whalesay cowsay boo-boo', we  

* create container
* load image docker/whalesay
* execute command 'cowsay' with parameter 'boo-boo' after image loaded.

Now, 'docker run docker-whale', after image loaded, the command 'usr/games/fortune -a | cowsay' will be executed, because we have added 'CMD ...' into our image

## Hello world in a container

Run a 'Hello world'  

```
docker run ubuntu /bin/echo 'Hello world'
```

Run an interactive container,  

* -t flag assigns a pseudo-tty or terminal inside the new container.
* -i flag allows you to make an interactive connection by grabbing the standard in (STDIN) of the container.  

```
docker run -t -i ubuntu /bin/bash
```

Start a daemonized Hello world,
* -d flag runs the container in the background  

```
docker run -d ubuntu /bin/sh -c "while true; do echo hello world; sleep 1; done"
```

Check the output from daemonized 'Hello word'  

```
docker ps

CONTAINER ID        IMAGE               COMMAND                  CREATED             STATUS              PORTS               NAMES
43584ee677db        ubuntu              "/bin/sh -c 'while tr"   4 seconds ago       Up 3 seconds                            cocky_northcutt

docker logs cocky_northcutt
```

Stop the running container  

```
docker stop cocky_northcutt
```

## Run a simple application

* Useful commands  

```
docker --help
docker attach --help
```

* Running a web application in Docker, -P flag to the docker run command Docker mapped any ports exposed in our image to our host.  

```
docker run -d -P training/webapp python app.py
docker ps -l
===
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS                     NAMES
80d23f8c0413        training/webapp     "python app.py"     20 seconds ago      Up 19 seconds       0.0.0.0:32768->5000/tcp   loving_mahavira
===
```
Note: -P default to use port 5000. In above example, -p 5000 that maps port 5000 inside the container to a high port. i.e. in this example map to 32768.  So you can now browse to port 32768 in a web browser to see the application

* Viewing the web application’s logs  

```
docker logs -f loving_mahavira
```

* Looking at our web application container’s processes  

```
docker top loving_mahavira
```

* Inspecting our web application container, you will see json output  

```
docker inspect loving_mahavira
===
[
    {
        "Id": "80d23f8c0413e07bc6241473bc2a0072319d8da8767ba848aa5b11156818cb26",
        "Created": "2016-11-02T13:17:50.620794499Z",
        "Path": "python",
        "Args": [
            "app.py"
        ],
        "State": {
            "Status": "running",
            ...
===

docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' loving_mahavira
===
172.17.0.2
===
```

* Stop, Restart, Remove our web application  

```
docker stop loving_mahavira
docker start loving_mahavira
docker rm loving_mahavira
```

## Network containers

### Name a container
You name your container by using the --name flag, for example launch a new container called web  

```
docker run -d -P --name web training/webapp python app.py
```

### Launch a container on the default network  

```
docker network ls
docker run -itd --name=networktest ubuntu
docker network inspect bridge
===
[
    {
        "Name": "bridge",
        "Id": "f7ab26d71dbd6f557852c7156ae0574bbf62c42f539b50c8ebde0f728a253b6f",
        "Scope": "local",
        "Driver": "bridge",
        "IPAM": {
            "Driver": "default",
            "Config": [
                {
                    "Subnet": "172.17.0.1/16",
                    "Gateway": "172.17.0.1"
                }
                ...
===
```

You can also disconnect a container from network  

```
docker network disconnect bridge networktest
```

Create your own bridge network, inspect it  

```
docker network create -d bridge my-bridge-network
docker network inspect my-bridge-network
```

Add containers to a network  

```
docker run -d --network=my-bridge-network --name db training/postgres
```

Connect container to a network  

```
docker network connect bridge db
```

## Data Volume

Adding a data volume, a directory /webapp will be created in container.  

```
docker run -d -P --name web -v /webapp training/webapp python app.py

```

Locate a Volume  

```
docker inspect web
===
"Mounts": [
    {
        "Name": "a12ac8ab3e202740494b6d854f2ac33b6bc26a827462383e0fd50e44f0d69007",
        "Source": "/var/lib/docker/volumes/a12ac8ab3e202740494b6d854f2ac33b6bc26a827462383e0fd50e44f0d69007/_data",
        "Destination": "/webapp",
        "Driver": "local",
        "Mode": "",
        "RW": true,
        "Propagation": ""
    }
],
===
```

Mount a host directory as a data volume  

```
docker run -d -P --name web -v ~/tmp:/webapp training/webapp python app.py
```
Using the -v flag you can also mount a directory from your Docker engine’s host into a container. this will create a shared directory between you host machine and container.

The container-dir must always be an absolute path such as /src/docs

### Mount a host file as a data volume  

```
docker run --rm -it -v ~/.bash_history:/root/.bash_history ubuntu /bin/bash
```
This will drop you into a bash shell in a new container, you will have your bash history from the host and when you exit the container, the host will have the history of the commands typed while in the container.

### Creating and mounting a data volume container
Create a new named container with a volume to share, the container is just created, not running yet.  

```
docker create -v /dbdata --name dbstore training/postgres /bin/true
```

Create new containers and running it with reused data volume  

```
docker run -d --volumes-from dbstore --name db1 training/postgres
docker run -d --volumes-from dbstore --name db2 training/postgres
```

### Backup, restore, migrate, or remove data volumes  

```
docker run --rm --volumes-from dbstore -v $(pwd):/backup ubuntu tar cvf /backup/backup.tar /dbdata

docker run -v /dbdata --name dbstore2 ubuntu /bin/bash

docker run --rm --volumes-from dbstore2 -v $(pwd):/backup ubuntu bash -c "cd /dbdata && tar xvf /backup/backup.tar --strip 1"

docker run --rm -v /foo -v awesome:/bar busybox top
```

### Be careful

Multiple containers can also share one or more data volumes. However, multiple containers writing to a single shared volume can cause data corruption. Make sure your applications are designed to write to shared data stores.

## Reference

https://docs.docker.com/
