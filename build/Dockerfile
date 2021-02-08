FROM fedora:32

RUN yum update -y
RUN yum install -y git
RUN git clone https://github.com/iphands/numclass.git && cd numclass/src/py
WORKDIR /numclass/src/py
RUN  pip install -r requirements.txt
WORKDIR /numclass/src/py/keras
CMD [ "/usr/bin/python3.8", "./main.py" ]
