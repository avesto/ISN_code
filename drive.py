f#Pour les arguments de ligne de commande 
import argparse

#décoder les images des caméras
import base64

#Sauvegarder avec le temps + jour d'enregistrement
from datetime import datetime

#lire dans des dossiers et créer des fichiers
import os

#Opérations sur dossiers de haut niveau
import shutil

#Maths maths maths
import numpy as np

#serveur en tps réel
import socketio

#maniment des évenements de reseaux
import eventlet

#Interface/porte du serveur web 
import eventlet.wsgi

#manipulaiton d'images
from PIL import Image

#framework web
from flask import Flask

#jouer avec l'ES
from io import BytesIO

#charger notre modèle
from keras.models import load_model

#une simple classe utilitaire
import utils








"""
Déja, pourquoi des librairies serveurs?
Parce que c'est ainsi que l'achitecture 
de Udacity est faite: le logiciel agit comme 
un serveur, et les scripts agissent comme un 
client, ils communiquent entre eux via une 
interface réseau
"""






#On commence par initialiser notre serveur
sio = socketio.Server()



#Notre webapp Falsk (CF ce qui à été dit au dessus)
app = Flask(__name__)



#on initialise nos modèles d'images et d'array comme nuls, on commence à zéro
model = None
prev_image_array = None



#Réglage de la vitesse max et min du véhicule...
MAX_SPEED = 25
MIN_SPEED = 10



#...et définition d'une limite de vitesse
speed_limit = MAX_SPEED





#mise en place d'un enregistreur d'évenèment pour le serveur
@sio.on('telemetry') #Fonction de télémétrie: récolte des données
def telemetry(sid, data):
    if data:

        # La position angulaire actuelle du volant
        steering_angle = float(data["steering_angle"])


        # L'accélération actuelle du véhicule ( pas pareil que la vitesse!!!!)
        throttle = float(data["throttle"])


        # Vitesse actuelle ( != accélération)
        speed = float(data["speed"])


        # Image de la caméra du centre actuelle
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        


        try:
            image = np.asarray(image)       # depuis l'image vers l'array numpy
            image = utils.preprocess(image) # Application de pre process sur l'image
            image = np.array([image])       # Le modèle attend un array numpy

            # Prédire un angle de rotation du volant
            steering_angle = float(model.predict(image, batch_size=1))
        



            """
             Baisser l'accélération quand la vitesse augemente 
             Si la vitesse dépasse la limite, on est en pente
             faire en sorte qu'on ralentisse en premier puis qu'on retourne a la vitesse max originale
            """
        
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # Ralentir
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2 




            #Affichage des différents paramètres
            print('{} {} {}'.format(steering_angle, throttle, speed))
            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)k



        #Sauvegarder la frame avec une nomenclature adaptée (merci datetime)
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        
        sio.emit('manual', data={}, skip_sid=True)



#Fonction permetttant la connetion au serveur
@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)



#Fonction permettant l'envoi des données vers le serveurs (données étant les paramètres de la fonction)
def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)






#Arguments de ligne de commande: afficher des informations dans l'invite pour l'utilisateur
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conduite autonome')
    parser.add_argument(
        'model',
        type=str,
        help='Destination vers le model h5. le modèle devrait etre dans la meme destination'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Destination vers le dossier contenant les images. C est la que les images seront sauvegardées'
    )
    args = parser.parse_args()

    #charger notre modèle 
    model = load_model(args.model)

    if args.image_folder != '':
        print("Création du dossier image dans {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("ENREGISTREMENT...")
    else:
        print("CET ESSAI N EST PAS ENREGISTRE...")




    # Routines de Flask & serveur/interface réseau
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)