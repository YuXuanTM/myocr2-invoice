FROM python:3.9.20
WORKDIR /usr/src/app
COPY . /usr/src/app
RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m pip install paddlepaddle-gpu==2.6.1.post117 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install paddleocr -i https://pypi.tuna.tsinghua.edu.cn/simple
EXPOSE 5000
CMD ["python", "main.py"]