---
title: "lightsail을 이용한 django 프로젝트 배포"
date: 2020-09-01
categories:
  - WEB
---


이번 한이음 프로젝트에서 개인적으로 웹을 만들게되었다.

한이음에서는 클라우드 서비스를 지원해주는데,  AWS의 lightsail을 사용하기로 했다.

일단 lightsail의 인스턴스를 받는것은 다른 블로그에도 잘 설명되있으니   
블루프린트로 ubuntu를 선택했다고 가정하고 시작한다.   


# 1. Gunicorn 설치
Gunicorn을 설치해 보자.

※ Gunicorn은 로컬환경에 설치할 필요는 없다. 서버환경에서만 Gunicorn을 설치하도록 하자.

Gunicorn은 가상환경 모드에서 다음처럼 pip을 이용하여 설치하면 된다.   

***
> (venv) ubuntu@ip-172-26-12-247:~/projects/my_site_prj$ pip install gunicorn   

***




잘 설치되는 것을 확인할 수 있다.

Gunicorn 테스트
Gunicorn이 정상적으로 동작하는지 다음처럼 실행해 보자.
   
***
>(venv) ubuntu@ip-172-26-12-247:~$ cd ~/projects/my_site_prj/   

>(venv) ubuntu@ip-172-26-12-247:~/projects/my_site_prj$ gunicorn --bind 0:8000 my_site_prj.wsgi:application   

***


먼저 /home/ubuntu/projects/my_site_prj 디렉터리로 이동한 후에 gunicorn --bind 0:8000 my_site_prj.wsgi:application 와 같이 수행한다. 
   
   
--bind 0:8000의 의미는 8000번 포트로 WSGI 서버를 수행한다는 의미이고  
my_site_prj.wsgi:application 은 WSGI 서버가 호출하는 WSGI 어플리케이션은 
my_site_prj/wsgi.py 파일의 application 이라는 의미이다.



서버가 오류없이 잘 시작되는 것을 확인할 수 있을 것이다.

하지만 웹브라우저로 다음 URL에 접속해 보면 레이아웃이 엉망이다


이렇게 보이는 이유는 Gunicorn이 정적(static)파일들을 해석하지 못하기 때문이다.    
정적파일로 bootstrap.min.css, bootstrap.min.js, style.css등 많은 정적파일을 필요로 한다
   
Gunicorn은 동적페이지 요청만 처리할 수 있기 때문에 위와 같이 표시되는 것이다.


일단 Gunicorn이 정상동작 되는것을 확인했으므로 Ctrl-C 를 입력하여 Gunicorn서버를 중지하도록 하자.
   
   
# 2. Gunicorn 소켓   

WSGI 서버인 Gunicorn은 위에서 테스트한 것과 같이 포트를 이용하여 서버를 띄울수도 있지만   
Unix계열 시스템에서는 포트로 서비스하는 방식 보다는 유닉스 소켓(Unix Socket)을 이용하는 방법을 더 선호한다.

이번에는 Gunicorn을 소켓으로 서비스하는 방법에 대해서 알아보자.

Gunicorn을 다음처럼 실행해 보자.
   
   
***
(venv) ubuntu@ip-172-26-12-247:~/projects/my_site_prj$ gunicorn --bind unix:/tmp/gunicorn.sock my_site_prj.wsgi:application   
[2020-04-17 01:14:51 +0000] [32392] [INFO] Starting gunicorn 20.0.4   
[2020-04-17 01:14:51 +0000] [32392] [INFO] Listening at: unix:/tmp/gunicorn.sock (32392)   
[2020-04-17 01:14:51 +0000] [32392] [INFO] Using worker: sync   
[2020-04-17 01:14:51 +0000] [32395] [INFO] Booting worker with pid: 32395   

***
포트방식으로 실행했을 때와 다른점은 --bind unix:/tmp/gunicorn.sock 부분이다. 
기존 포트방식(--bind 0:8000) 대신 소켓방식(--bind unix:/tmp/gunicorn.sock)으로 변경되었다.
 소켓파일은 /tmp/gunicorn.sock 이라는 파일로 생성된다.

※ 이렇게 소켓방식으로 Gunicorn 서버를 실행할 경우에는 단독으로 Gunicorn을 테스트할 수 없다.
 Nginx와 같은 웹서버에서 이 소켓파일로 WSGI 서버에 접속하도록 설정해야만 테스트가 가능하다.

# 3. Gunicorn 서비스
이번에는 Gunicorn을 서비스로 등록해 보자. 
서비스로 등록하는 이유는 Gunicorn의 시작, 중지를 쉽게 하기 위함이고 
또 AWS 서버가 재 시작될 때 자동으로 Gunicorn을 실행해 주기 위해서이다.

Gunicorn을 서비스로 등록하기 위해서는 환경변수 파일과 서비스 파일을 작성해야 한다.

서비스 파일
서비스 파일은 /etc/systemd/system/ 디렉터리에 gunicorn.service이라는 이름으로 생성한다.

※ 이 디렉터리는 시스템 디렉터리이기 때문 sudo vi gunicorn.service 처럼 관리자(sudo) 권한으로 파일을 생성해 주어야 한다.

[/etc/systemd/system/gunicorn.service]
   
   
***
[Unit]
Description=gunicorn daemon   
After=network.target   

[Service]   
User=ubuntu   
Group=ubuntu   
WorkingDirectory=/home/ubuntu/Web-django   
ExecStart=/home/ubuntu/Web-django/venv/bin/gunicorn \   
        --workers 2 \   
        --bind unix:/tmp/gunicorn.sock \   
        my_site_prj.wsgi:application   
[Install]   
WantedBy=multi-user.target   

***


위 파일에서 사용된 EnvironmentFile 항목이 위에서 작성한 환경변수 파일을 불러오는 부분이다. --worker 2의 의미는 Gunicorn 프로세스 갯수를 2개로 사용하라는 의미이다.

# 4. 서비스 실행과 등록
파일이 생성되었으면 다음처럼 서비스를 실행해 보자. 서비스 실행은 역시 관리자(sudo)로 실행해야 한다.
   
***
>sudo systemctl start gunicorn.service   

***
   
잘 실행되었는지 확인하기위해 다음처럼 sudo systemctl status gunicorn.service 명령을 실행해 보자.
 
   
***
>(venv) ubuntu@ip-172-26-12-247:/etc/systemd/system$ sudo systemctl status gunicorn.service   

● gunicorn.service - gunicorn daemon   
   Loaded: loaded (/etc/systemd/system/gunicorn.service; disabled; vendor preset: enabled)   
   Active: active (running) since Thu 2020-04-23 12:12:27 UTC; 1s ago   
 Main PID: 26513 (gunicorn)   
    Tasks: 3 (limit: 547)   
   CGroup: /system.slice/gunicorn.service   
           ├─26513 /home/ubuntu/venvs/my_site_prj/bin/python3 /home/ubuntu/venvs/my_site_prj/bin/gunicorn --workers 2 --bind unix:/tmp/gunicorn.sock my_site_prj.wsgi:application   
           ├─26534 /home/ubuntu/venvs/my_site_prj/bin/python3 /home/ubuntu/venvs/my_site_prj/bin/gunicorn --workers 2 --bind unix:/tmp/gunicorn.sock my_site_prj.wsgi:application   
           └─26536 /home/ubuntu/venvs/my_site_prj/bin/python3 /home/ubuntu/venvs/my_site_prj/bin/gunicorn --workers 2 --bind unix:/tmp/gunicorn.sock my_site_prj.wsgi:application   
   
Apr 23 12:12:27 ip-172-26-12-247 systemd[1]: Started gunicorn daemon.   
Apr 23 12:12:28 ip-172-26-12-247 gunicorn[26513]: [2020-04-23 12:12:28 +0000] [26513] [INFO] Starting gunicorn 20.0.4   
Apr 23 12:12:28 ip-172-26-12-247 gunicorn[26513]: [2020-04-23 12:12:28 +0000] [26513] [INFO] Listening at: unix:/tmp/gunicorn.sock (26513)   
Apr 23 12:12:28 ip-172-26-12-247 gunicorn[26513]: [2020-04-23 12:12:28 +0000] [26513] [INFO] Using worker: sync   
Apr 23 12:12:28 ip-172-26-12-247 gunicorn[26513]: [2020-04-23 12:12:28 +0000] [26534] [INFO] Booting worker with pid: 26534   
Apr 23 12:12:28 ip-172-26-12-247 gunicorn[26513]: [2020-04-23 12:12:28 +0000] [26536] [INFO] Booting worker with pid: 26536   
(venv) ubuntu@ip-172-26-12-247:/etc/systemd/system$   

***
   
정상적으로 실행되었다면 아마도 위와 비슷한 메시지들이 출력될 것이다.
   
※ 만약 위와 같은 메시지 대신 오류가 확인되면 /var/log/syslog 파일에서 오류가 난 원인을 확인하고 수정해야 한다.

마지막으로 AWS 서버가 재시작될 때 자동으로 Gunicorn이 실행되기 위해서 다음처럼 enable 옵션을 이용하여 서비스로 등록하도록 한다.
   
***
>sudo systemctl enable gunicorn.service   

***

# 5. Nginx 에 대하여

Nginx는 요새 급성장하고 있는 웹서버로 아파치(Apache)를 대신할 차세대 웹서버로 주목받고 있다. 
Nginx는 높은 성능을 위해서 개발된 웹서버로 점점 사용자가 증가하는 추세이며 특히 파이썬 웹 프레임워크인 장고나 플라스크등에서 주로 사용되는 서버이다.
 또한 Nginx를 사용하기 위한 설정도 무척 간단하여 쉽게 사용할수 있다.

# 6. Nginx 설치

Nginx를 다음과 같이 관리자 권한으로 설치하자.
   
***
>(venv) ubuntu@ip-172-26-12-247:~/projects/my_site_prj$ sudo apt install nginx   

***
대략 10초 내외로 설치가 될 것이다.

# 7. Nginx 설정

Nginx를 설치한 후 동적페이지 요청 발생시 WSGI 서버를 호출하도록 설정해 보자.


먼저 /etc/nginx/sites-available/ 디렉터리로 이동한다.
   
***
>(venv) ubuntu@ip-172-26-12-247:~/projects/my_site_prj$ cd /etc/nginx/sites-available/

>(venv) ubuntu@ip-172-26-12-247:/etc/nginx/sites-available$   

***

/etc/nginx/sites-available 디렉터리는 Nginx의 설정파일들이 위치한 디렉터리이다. 최초 설치시에는 deafult라는 설정파일만 존재한다.

그리고 파이보 시스템에 대한 Nginx의 설정파일을 다음과 같이 작성한다.


※ 시스템 디렉터리이므로 sudo vi mysite (또는 sudo nano mysite) 와 같이 관리자 권한으로 작성해야 한다.
   
***
>(venv) ubuntu@ip-172-26-12-247:/etc/nginx/sites-available$ sudo vi mysite   

***

그리고 mysite 파일을 다음과 같이 편집한다.

[/etc/nginx/sites-available/my_site_prj]   
   
   
***   

server {   
        listen 80;   
        
        server_name 52.78.8.100;   
        
        location = /favicon.ico { access_log off; log_not_found off; }   
        location /static {   
                alias /home/ubuntu/Web-django/static;   
        }   
        location / {   
                include proxy_params;    
                proxy_pass http://unix:/tmp/gunicorn.sock;   
        }   
}   

***
   
listen 80 은 웹서버를 80포트로 서비스 한다는 의미이다. HTTP 프로토콜의 기본포트는 80이다.    
따라서 이제 http://blahblah:8000/ 대신 포트를 생략하여 http://blahblah/ 처럼 웹브라우저에서 호출 할 수 있을 것이다.   
 server_name 에는 여러분의 고정아이피를 등록하도록 하자.      

location /static 은 정적파일에 대한 설정으로 웹브라우저에서 /static으로 시작되는 URL요청은
 Nginx가 /home/ubuntu/Web-django/static 디렉터리의 파일을 읽어서 처리한다는 설정이다.


location /은 location /static 에서 설정한 것 이외의 모든 요청은 Gunicorn이 처리하도록 하는 설정이다.
 proxy_pass 는 이전 장에서 설정했던 Gunicorn의 소켓파일 경로를 위와 같은 형식으로 입력해 준다.


이와 같은 설정을 통해 /static 으로 시작되는 URL은 Nginx가 처리하고 나머지 URL에 대해서는 Gunicorn이 처리하게 된다.


이제 작성한 mysite 파일을 Nginx가 환경파일로 읽을 수 있도록 설정해 주어야 한다.


다음처럼 /etc/nginx/sites-enabled/ 디렉터리로 이동한다.
   
***
>(venv) ubuntu@ip-172-26-12-247:/etc/nginx/sites-available$ cd /etc/nginx/sites-enabled/   
>(venv) ubuntu@ip-172-26-12-247:/etc/nginx/sites-enabled$   

***
   
sites-enabled 디렉터리는 site-available 디렉터리에 있는 설정파일 여러개중 활성화시키고 싶은 설정파일을 관리하는 디렉터리이다. 
ls 명령을 수행하면 현재 default 설정파일만 링크되어 있는 것을 확인할 수 있다.


이제 이 default링크는 삭제하고 mysite파일을 링크하도록 변경해 주어야 한다. 먼저 다음처럼 default링크를 삭제하자.   
***
>(venv) ubuntu@ip-172-26-12-247:/etc/nginx/sites-enabled$ sudo rm default   

***

그리고 다음처럼 이미 작성한 mysite 파일을 링크한다.
   
***
>(venv) ubuntu@ip-172-26-12-247:/etc/nginx/sites-enabled$ sudo ln -s /etc/nginx/sites-available/my_site_prj   

***

그리고 ls명령을 수행해 보면 default는 사라지고 mysite 링크만 남아있는 것을 확인할 수 있을 것이다.


# 8. Nginx 실행

Nginx는 설치할때 자동으로 실행된다. 따라서 Nginx에 변경된 설정을 적용하기 위해 다음과 같이 Nginx를 재시작해 주어야 한다.
   
***
>(venv) ubuntu@ip-172-26-12-247:/etc/nginx/sites-enabled$ sudo systemctl restart nginx   

***


이제 접속을 해보면

    
    ![image](https://user-images.githubusercontent.com/49622935/91743817-61e1f500-ebf3-11ea-9c4d-d17e53a56c22.png)   
    
    잘 구동되는것을 알 수 있다


아직 초기버전이라 이 사이트의 기능과 완성된 레이아웃은 나중에 기록될 예정이다


다음 포스트로는 로컬에서 서버에 접속하는 방법과 보안의 강화 등을 포스트해보겠다


