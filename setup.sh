#!/bin/bash

# Instala Nginx y el módulo RTMP
sudo apt install nginx libnginx-mod-rtmp

# Agrega la configuración RTMP al archivo nginx.conf
if ! grep -q "rtmp {" /etc/nginx/nginx.conf; then
    echo "rtmp {
    server {
        listen 1935;
        application live {
        live on;
        allow publish all;
        allow play all;
        record off;
        }
    }
    }" | sudo tee -a /etc/nginx/nginx.conf

# Reinicia Nginx para aplicar los cambios
sudo service nginx restart
else
    echo "La configuración RTMP ya está presente en nginx.conf. No se realizaron cambios."
fi

# Crear un entorno virtual y activarlo
python3 -m venv .venv
source .venv/bin/activate

# Instalar paquetes necesarios
pip install -r requirements.txt

# Ejecutar el script en python
python smm_app/smm_web/web_flask.py

# Salir del entorno
deactivate